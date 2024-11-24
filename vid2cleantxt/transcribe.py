#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe.py - transcribe a video file using a pretrained ASR model such as wav2vec2 or whisper

Usage:
    vid2cleantxt.py --video <video_file> --model <model_id> [--out <out_dir>] [--verbose] [--debug] [--log <log_file>]

Tips for runtime:

- use the default model to start. If issues try "facebook/wav2vec2-base-960h"
- if model fails to work or errors out, try reducing the chunk length with --chunk-length <int>
"""

__author__ = "Peter Szemraj"

import argparse
import gc
import os
import sys
from os.path import dirname, join

sys.path.append(dirname(dirname(os.path.abspath(__file__))))


import logging
import re

logging.basicConfig(level=logging.INFO, filename="LOGFILE_vid2cleantxt_transcriber.log", format="%(asctime)s %(message)s")

import argparse
import math
import shutil
import time
import warnings

import librosa
import pandas as pd
import torch
import transformers
from tqdm.auto import tqdm
from transformers import (
    HubertForCTC,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    WavLMForCTC,
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

#  filter out warnings that pretend transfer learning does not exist
warnings.filterwarnings("ignore", message="Some weights of")
warnings.filterwarnings("ignore", message="initializing BertModel")
transformers.utils.logging.set_verbosity(40)

from vid2cleantxt.v2ct_utils import load_spacy_models

load_spacy_models()

from vid2cleantxt.audio2text_functions import (
    corr,
    create_metadata_df,
    get_av_fmts,
    prep_transc_pydub,
    quick_keys,
    setup_out_dirs,
    trim_fname,
)
from vid2cleantxt.v2ct_utils import (
    NullIO,
    check_runhardware,
    create_folder,
    digest_txt_directory,
    find_ext_local,
    get_timestamp,
    load_spacy_models,
    move2completed,
    torch_validate_cuda,
)


def format_time_for_srt(seconds):
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm)
    
    :param seconds: Time in seconds (float)
    :return: Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    seconds = int(seconds)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def parse_transcription_with_timestamps(transcription):
    """
    Parse Whisper transcription with timestamps into segments
    """
    segments = []
    import re
    
    # Updated pattern to handle more timestamp formats
    pattern = r'<\|(\d+\.\d+)\|>|[\[\(](\d+\.\d+)[\]\)]'
    
    # Split transcription into chunks with timestamps
    parts = re.split(pattern, transcription)
    
    # Extract all timestamps (handling both capture groups)
    timestamps = []
    matches = re.finditer(pattern, transcription)
    for match in matches:
        # Get whichever group matched (group 1 or 2)
        ts = match.group(1) or match.group(2)
        timestamps.append(float(ts))
    
    # Clean and process segments
    current_text = ""
    for i in range(1, len(parts)-1, 2):
        text = parts[i].strip()
        if text and not text.startswith('<|') and not text.startswith('['):
            if i//2 < len(timestamps)-1:
                segments.append({
                    'start': timestamps[i//2],
                    'end': timestamps[i//2 + 1],
                    'text': text
                })
    
    return segments

def load_whisper_modules(
    hf_id: str, task: str = "transcribe", chunk_length: int = 30, language: str = None
):
    """
    load_whisper_modules - load the whisper modules from huggingface
    
    Parameters
    ----------
    hf_id : str
        The Huggingface model ID
    task : str, optional
        Task to perform, by default "transcribe"
    chunk_length : int, optional
        Length of audio chunks in seconds, by default 30
    language : str, optional
        Language code (e.g. 'en', 'fr'). If None, will auto-detect language
    """
    processor = WhisperProcessor.from_pretrained(hf_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        hf_id,
        use_cache=True
    )

    processor.feature_extractor.chunk_length = chunk_length
    
    # Only set forced decoder IDs if language is specified
    if language:
        model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=language, task=task
        )
    else:
        # For auto-detection, don't set forced decoder IDs
        model.config.forced_decoder_ids = None

    return processor, model


def load_wav2vec2_modules(hf_id: str):
    """
    load_transcription_objects - load the transcription objects from huggingface

    Parameters
    ----------
    hf_id : str, the id of the model to load on huggingface, for example: "facebook/wav2vec2-base-960h" or "facebook/hubert-large-ls960-ft"

    Returns
    -------
    tokenizer : transformers.Wav2Vec2Processor, the tokenizer object
    model: transformers.Wav2Vec2ForCTC, the model object. For specialised models, this is a specialised object such as HubertForCTC
    """

    tokenizer = Wav2Vec2Processor.from_pretrained(
        hf_id
    )  # use wav2vec2processor for tokenization always
    if "wavlm" in hf_id.lower():
        # for example --model "patrickvonplaten/wavlm-libri-clean-100h-large"
        print(f"Loading wavlm model - {hf_id}")
        model = WavLMForCTC.from_pretrained(hf_id)
    elif "hubert" in hf_id.lower():
        print(f"Loading hubert model - {hf_id}")
        model = HubertForCTC.from_pretrained(
            hf_id
        )  # for example --model "facebook/hubert-large-ls960-ft"
    else:
        # for example --model "facebook/wav2vec2-large-960h-lv60-self"
        print(f"Loading wav2vec2 model - {hf_id}")
        model = Wav2Vec2ForCTC.from_pretrained(hf_id)
    logging.info(f"Loaded model {hf_id} from huggingface")
    return tokenizer, model


def wav2vec2_islarge(model_obj):
    """
    wav2vec2_check_size - compares the size of the passed model object, and whether
    it is in fact a wav2vec2 model. this is because the large model is a special case and
    uses an attention mechanism that is not compatible with the rest of the models

    https://huggingface.co/facebook/wav2vec2-base-960h

    Parameters
    ----------
    model_obj : transformers.Wav2Vec2ForCTC, the model object to check

    Returns
    -------
    is_large, whether the model is the large wav2vec2 model or not
    """
    approx_sizes = {
        "base": 94396320,
        "large": 315471520,  # recorded by  loading the model in known environment
    }
    if isinstance(model_obj, HubertForCTC):
        logging.info("HubertForCTC is not a wav2vec2 model so not checking size")
        return False
    elif not isinstance(model_obj, Wav2Vec2ForCTC):
        warnings.warn(
            message="Model is not a wav2vec2 model - this function is for wav2vec2 models only",
            category=None,
            stacklevel=1,
        )
        return (
            False  # not a wav2vec2 model - return false so it is handled per standard
        )

    np_proposed = model_obj.num_parameters()

    dist_from_base = abs(np_proposed - approx_sizes.get("base"))
    dist_from_large = abs(np_proposed - approx_sizes.get("large"))
    return True if dist_from_large < dist_from_base else False


def save_transc_results(
    out_dir,
    vid_name: str,
    ttext: str,
    mdata: pd.DataFrame,
    srt_entries: list = None,
    verbose=False,
):
    """
    save_transc_results - save the transcribed text to a file and a metadata file
    """
    storage_locs = setup_out_dirs(out_dir)  # create and get output folders
    out_p_tscript = storage_locs.get("t_out")
    out_p_metadata = storage_locs.get("m_out")
    header = f"{trim_fname(vid_name)}_vid2txt_{get_timestamp()}"
    
    # Save regular transcript
    _t_out = join(out_p_tscript, f"{header}_full.txt")
    with open(
        _t_out,
        "w",
        encoding="utf-8",
        errors="ignore",
    ) as fo:
        fo.writelines(ttext)

    # Save SRT file if entries provided
    if srt_entries and len(srt_entries) > 0:
        srt_filename = f"{header}.srt"
        # Save SRT in the main transcriptions directory
        srt_filepath = join(dirname(out_p_tscript), srt_filename)
        logging.info(f"Saving {len(srt_entries)} SRT entries to {srt_filepath}")
        with open(srt_filepath, "w", encoding="utf-8") as srt_file:
            srt_file.writelines(srt_entries)
        if verbose:
            print(f"SRT file saved to: {srt_filepath}")
        logging.info(f"SRT file saved to: {srt_filepath}")
    else:
        logging.warning("No SRT entries to save")

    mdata.to_csv(join(out_p_metadata, f"{header}_metadata.csv"), index=False)

    if verbose:
        print(
            f"Saved transcript and metadata to: {out_p_tscript} \n and {out_p_metadata}"
        )

    logging.info(
        f"Saved transcript and metadata to: {out_p_tscript} and {out_p_metadata}"
    )


def load_wav2vec2_alignment_model(language_code: str = "en"):
    """
    Load wav2vec2 model for forced alignment
    """
    try:
        if not language_code:
            language_code = "en"
            logging.warning("No language specified, defaulting to English for alignment")
            
        if language_code == "en":
            model_id = "facebook/wav2vec2-large-960h-lv60-self"
        else:
            model_id = f"facebook/wav2vec2-large-xlsr-53-{language_code}"
        
        logging.info(f"Loading alignment model: {model_id}")
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        return model, processor
    except Exception as e:
        logging.error(f"Failed to load alignment model: {str(e)}")
        return None, None

def detect_speech_segments(audio_input, sample_rate=16000):
    """
    Use VAD to detect speech segments
    """
    try:
        import torch
        
        # Load VAD model with explicit trust
        model = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            trust_repo=True,
            force_reload=False
        )
        
        # Convert audio to tensor if needed
        if not isinstance(audio_input, torch.Tensor):
            audio_input = torch.tensor(audio_input)
        
        # The model itself is the first element of the tuple
        vad_model = model[0]
        
        # Define chunk size based on sample rate
        chunk_size = 512 if sample_rate == 16000 else 256
        
        # Process audio in chunks
        speech_probs = []
        for i in range(0, len(audio_input), chunk_size):
            chunk = audio_input[i:i + chunk_size]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
            
            # Get speech probability for chunk
            with torch.no_grad():
                prob = vad_model(chunk, sample_rate)
                speech_probs.append(prob.item())
        
        # Convert probabilities to speech segments
        threshold = 0.5  # Adjust this threshold as needed
        speech_mask = [prob > threshold for prob in speech_probs]
        
        # Find continuous speech segments
        segments = []
        in_speech = False
        start_time = 0
        chunk_duration = chunk_size / sample_rate
        
        for i, is_speech in enumerate(speech_mask):
            if is_speech and not in_speech:
                # Start of speech segment
                start_time = i * chunk_duration
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                end_time = i * chunk_duration
                segments.append((start_time, end_time))
                in_speech = False
        
        # Handle case where audio ends during speech
        if in_speech:
            end_time = len(speech_mask) * chunk_duration
            segments.append((start_time, end_time))
        
        # Merge segments that are close together
        merged_segments = []
        if segments:
            current_start, current_end = segments[0]
            min_gap = 0.3  # Minimum gap in seconds to consider segments separate
            
            for start, end in segments[1:]:
                if start - current_end <= min_gap:
                    # Merge segments
                    current_end = end
                else:
                    # Save current segment and start new one
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = start, end
            
            # Add last segment
            merged_segments.append((current_start, current_end))
                
        logging.info(f"Detected {len(merged_segments)} speech segments")
        return merged_segments
    except Exception as e:
        logging.error(f"VAD failed: {str(e)}")
        logging.error(f"VAD error details: {type(e).__name__}")
        logging.error(f"VAD error full details: {e}")
        return None

def forced_alignment(audio_input, text, align_model, align_processor, device="cuda"):
    """
    Perform forced alignment using wav2vec2's CTC alignment
    """
    try:
        # Get audio features
        inputs = align_processor(audio_input, sampling_rate=16000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        # Get phoneme probabilities
        with torch.no_grad():
            outputs = align_model(input_values)
            logits = outputs.logits
            
            # Get CTC probabilities
            log_probs = torch.nn.functional.log_softmax(logits.squeeze(0), dim=-1)
            
            # Process text into tokens
            words = text.lower().strip().split()
            
            # Calculate frame duration
            num_frames = log_probs.size(0)
            audio_duration = len(audio_input) / 16000
            frame_duration = audio_duration / num_frames
            
            # Initialize alignments
            word_alignments = []
            
            # Process each word
            for word_idx, word in enumerate(words):
                try:
                    # Get token ids for this word
                    token_ids = align_processor.tokenizer(word)['input_ids']
                    
                    # Skip empty tokens
                    if not token_ids:
                        continue
                        
                    # Get probabilities for each token
                    token_probs = []
                    for token_id in token_ids:
                        token_prob = log_probs[:, token_id]
                        token_probs.append(token_prob)
                    
                    # Stack probabilities
                    if token_probs:
                        token_probs = torch.stack(token_probs, dim=1)
                        
                        # Find best frame for each token
                        max_probs, max_frames = torch.max(token_probs, dim=0)
                        
                        # Get word boundaries
                        word_start_frame = torch.min(max_frames)
                        word_end_frame = torch.max(max_frames)
                        
                        # Convert to time
                        word_start = float(word_start_frame) * frame_duration
                        word_end = float(word_end_frame + 1) * frame_duration
                        
                        word_alignments.append((
                            int(word_start * 16000),  # Convert to samples
                            int(word_end * 16000)
                        ))
                    else:
                        # Fallback: estimate position
                        estimated_start = audio_duration * (word_idx / len(words))
                        estimated_end = audio_duration * ((word_idx + 1) / len(words))
                        
                        word_alignments.append((
                            int(estimated_start * 16000),
                            int(estimated_end * 16000)
                        ))
                        
                except Exception as e:
                    logging.warning(f"Failed to align word '{word}': {str(e)}")
                    # Fallback for this word
                    estimated_start = audio_duration * (word_idx / len(words))
                    estimated_end = audio_duration * ((word_idx + 1) / len(words))
                    
                    word_alignments.append((
                        int(estimated_start * 16000),
                        int(estimated_end * 16000)
                    ))
            
            if not word_alignments:
                raise ValueError("No valid alignments found")
            
            # Ensure timestamps are monotonically increasing
            for i in range(1, len(word_alignments)):
                if word_alignments[i][0] <= word_alignments[i-1][1]:
                    word_alignments[i] = (
                        word_alignments[i-1][1] + 1,
                        max(word_alignments[i][1], word_alignments[i-1][1] + int(0.1 * 16000))  # At least 100ms
                    )
            
            return word_alignments
            
    except Exception as e:
        logging.error(f"Alignment failed: {str(e)}")
        logging.error(f"Alignment error details: {type(e).__name__}")
        raise  # Re-raise to prevent fallback

def transcribe_video_whisper(
    model,
    processor,
    clip_directory,
    clip_name: str,
    chunk_dur: int = 30,
    chunk_max_new_tokens=444,
    temp_dir: str = "audio_chunks",
    manually_clear_cuda_cache=False,
    print_memory_usage=False,
    verbose=False,
    generate_srt=False,
    language=None,
) -> dict:
    """
    Transcribe video using Whisper + forced alignment for accurate timestamps
    """
    logging.info(f"Starting to transcribe {clip_name}")
    if verbose:
        print(f"Starting to transcribe {clip_name} @ {get_timestamp()}")
    ac_storedir = join(clip_directory, temp_dir)
    create_folder(ac_storedir)
    chunk_directory = prep_transc_pydub(
        clip_name, clip_directory, ac_storedir, chunk_dur, verbose=verbose
    )
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"transcribing on {device}")
    full_transc = []
    GPU_update_incr = (
        math.ceil(len(chunk_directory) / 2) if len(chunk_directory) > 1 else 1
    )

    # Initialize srt_entries list if generating SRT
    srt_entries = [] if generate_srt else None
    srt_counter = 1

    if generate_srt:
        logging.info("SRT generation enabled")
        logging.warning(
            "Note: SRT timestamps from Whisper may not be perfectly synced with speech. "
            "For precise lip-sync, consider using additional forced alignment tools like WhisperX"
        )

    model = model.to(device)
    pbar = tqdm(total=len(chunk_directory), desc="Transcribing video")

    # Load alignment model if generating SRT
    align_model = None
    align_processor = None
    if generate_srt:
        logging.info("Loading alignment model for accurate timestamps")
        align_model, align_processor = load_wav2vec2_alignment_model(language)
        if align_model:
            align_model = align_model.to(device)
        else:
            logging.warning("Failed to load alignment model, falling back to basic timestamps")

    # Process each chunk
    for i, audio_chunk in enumerate(chunk_directory):
        try:
            # Load audio
            audio_input, clip_sr = librosa.load(
                join(ac_storedir, audio_chunk), sr=16000
            )

            # Detect speech segments using VAD
            speech_segments = None
            if generate_srt:
                speech_segments = detect_speech_segments(audio_input)

            # Get Whisper transcription
            input_features = processor(
                audio_input,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)

            outputs = model.generate(
                input_features=input_features,
                max_new_tokens=chunk_max_new_tokens,
                task="transcribe",
                language=language if language else None,
                return_timestamps=True,
                output_hidden_states=False,
                return_dict_in_generate=True,
            )
            
            # Handle different output formats
            if hasattr(outputs, 'sequences'):
                transcription = processor.batch_decode(
                    outputs.sequences,
                    skip_special_tokens=True
                )[0]
            else:
                # Handle case where outputs is the sequence tensor directly
                transcription = processor.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )[0]
            
            logging.info(f"Transcription output: '{transcription}'")
            
            # If we have alignment model and speech segments, improve timestamps
            if generate_srt and align_model and speech_segments:
                try:
                    # Perform forced alignment
                    alignment = forced_alignment(
                        audio_input, 
                        transcription,
                        align_model,
                        align_processor,
                        device
                    )

                    if not alignment:
                        raise ValueError("Alignment failed to produce timestamps")

                    # Create SRT entries with aligned timestamps
                    words = transcription.split()
                    current_segment = []
                    current_start = None
                    max_words_per_segment = 10
                    
                    for word_idx, word in enumerate(words):
                        if word_idx < len(alignment):
                            word_start = (alignment[word_idx][0] / 16000)
                            word_end = (alignment[word_idx][1] / 16000)
                            
                            # Start new segment if needed
                            if current_start is None:
                                current_start = word_start
                            
                            current_segment.append(word)
                            
                            # Create SRT entry when segment is full or at last word
                            if len(current_segment) >= max_words_per_segment or word_idx == len(words) - 1:
                                srt_entry = (
                                    f"{srt_counter}\n"
                                    f"{format_time_for_srt(current_start)} --> {format_time_for_srt(word_end)}\n"
                                    f"{' '.join(current_segment)}\n\n"
                                )
                                srt_entries.append(srt_entry)
                                srt_counter += 1
                                
                                # Reset for next segment
                                current_segment = []
                                current_start = None
                
                    logging.info(f"Created {srt_counter-1} subtitle segments")

                except Exception as e:
                    logging.error(f"Error in forced alignment: {str(e)}")
                    raise  # Re-raise instead of falling back

            # Extract segments with timestamps
            segments = outputs['segments']

            # Calculate the chunk's start time
            chunk_start_time = i * chunk_dur

            # Adjust timestamps by adding the chunk's start time
            for segment in segments:
                start_time = segment['start'] + chunk_start_time
                end_time = segment['end'] + chunk_start_time
                text = segment['text']
                srt_entry = f"{srt_counter}\n{format_time_for_srt(start_time)} --> {format_time_for_srt(end_time)}\n{text.strip()}\n\n"
                srt_entries.append(srt_entry)
                srt_counter += 1

            # ... rest of the existing chunk processing code ...

        except Exception as e:
            logging.error(f"Error processing chunk {i}: {str(e)}")
            continue

        if device == "cuda" and manually_clear_cuda_cache:
            torch.cuda.empty_cache()

        pbar.update()

    pbar.close()
    logging.info("completed transcription")

    md_df = create_metadata_df()  # blank df with column names
    full_text = corr(" ".join(full_transc))
    md_df.loc[len(md_df), :] = [
        clip_name,
        len(chunk_directory),
        chunk_dur,
        (len(chunk_directory) * chunk_dur) / 60,  # duration in mins
        get_timestamp(),
        full_text,
        len(full_text),
        len(full_text.split(" ")),
    ]
    md_df.transpose(
        copy=False,
    )
    save_transc_results(
        out_dir=clip_directory,
        vid_name=clip_name,
        ttext=full_text,
        mdata=md_df,
        srt_entries=srt_entries,  # This will now contain the entries if generate_srt=True
        verbose=verbose,
    )

    shutil.rmtree(ac_storedir, ignore_errors=True)
    transc_res = {
        "audio_transcription": full_transc,
        "metadata": md_df,
    }

    if verbose:
        print(f"finished transcription of {clip_name} - {get_timestamp()}")
    logging.info(f"finished transcription of {clip_name} - {get_timestamp()}")

    # Clean up alignment model
    if align_model:
        del align_model
        if device == "cuda":
            torch.cuda.empty_cache()

    return transc_res


def transcribe_video_wav2vec(
    model,
    processor,
    clip_directory,
    clip_name: str,
    chunk_dur: int = 15,
    temp_dir: str = "audio_chunks",
    manually_clear_cuda_cache=False,
    print_memory_usage=False,
    verbose=False,
) -> dict:
    """
    transcribe_video_wav2vec - transcribe a video file using the wav2vec model

    :param model: the model object
    :param processor: the processor object
    :param clip_directory: the directory of the video file
    :param str clip_name: the name of the video file
    :param int chunk_dur: the duration of each chunk in seconds, default 15
    :param str temp_dir: the directory to store the audio chunks in. default "audio_chunks"
    :param bool manually_clear_cuda_cache: whether to manually clear the cuda cache after each chunk. default False
    :param bool print_memory_usage: whether to print the memory usage at set interval while transcribing. default False
    :param bool verbose: whether to print the transcribed text locations to the console. default False
    :return dict: a dictionary containing the transcribed text, the metadata
    """
    logging.info(f"Starting to transcribe {clip_name}")
    if verbose:
        print(f"Starting to transcribe {clip_name} @ {get_timestamp()}")
    ac_storedir = join(clip_directory, temp_dir)
    create_folder(ac_storedir)
    use_attn = wav2vec2_islarge(
        model
    )  # if they pass in a large model, use attention masking

    chunk_directory = prep_transc_pydub(
        clip_name, clip_directory, ac_storedir, chunk_dur, verbose=verbose
    )  # split the video into chunks
    gc.collect()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logging.info(f"transcribing on {device}")
    full_transc = []
    GPU_update_incr = (
        math.ceil(len(chunk_directory) / 2) if len(chunk_directory) > 1 else 1
    )
    model = model.to(device)
    pbar = tqdm(total=len(chunk_directory), desc="Transcribing video")
    for i, audio_chunk in enumerate(chunk_directory):

        # note that large-960h-lv60 has an attention mask of length of the input sequence, the base model does not
        if (i % GPU_update_incr == 0) and (GPU_update_incr != 0) and print_memory_usage:
            check_runhardware()  # check utilization
            gc.collect()
        try:
            audio_input, clip_sr = librosa.load(
                join(ac_storedir, audio_chunk), sr=16000
            )  # load the audio chunk @ 16kHz (wav2vec2 expects 16kHz)

            inputs = processor(
                audio_input, return_tensors="pt", padding="longest"
            )  # audio to tensor
            input_values = inputs.input_values.to(device)
            attention_mask = (
                inputs.attention_mask.to(device) if use_attn else None
            )  # set attention mask if using large model

            with torch.no_grad():
                if use_attn:
                    logits = model(input_values, attention_mask=attention_mask).logits
                else:
                    logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)  # get the predicted ids
            this_transc = processor.batch_decode(predicted_ids)
            this_transc = (
                "".join(this_transc) if isinstance(this_transc, list) else this_transc
            )
        except Exception as e:
            logging.warning(
                f"Error transcribing chunk {i} in {clip_name}"
            )
            logging.warning(e)
            warnings.warn(f"Error transcribing chunk {i} - see log for details   ")
            this_transc = ""

        full_transc.append(f"{this_transc}\n")

        del input_values
        del logits
        del predicted_ids
        if device == "cuda" and manually_clear_cuda_cache:
            torch.cuda.empty_cache()

        pbar.update()

    pbar.close()
    logging.info("completed transcription")

    md_df = create_metadata_df()  # makes a blank df with column names
    full_text = corr(" ".join(full_transc))
    md_df.loc[len(md_df), :] = [
        clip_name,
        len(chunk_directory),
        chunk_dur,
        (len(chunk_directory) * chunk_dur) / 60,
        get_timestamp(),
        full_text,
        len(full_text),
        len(full_text.split(" ")),
    ]
    md_df.transpose(
        copy=False,
    )
    save_transc_results(
        out_dir=clip_directory,
        vid_name=clip_name,
        ttext=full_text,
        mdata=md_df,
        verbose=verbose,
    )

    shutil.rmtree(ac_storedir, ignore_errors=True)
    transc_res = {
        "audio_transcription": full_transc,
        "metadata": md_df,
    }

    if verbose:
        print(f"finished transcription of {clip_name} - {get_timestamp()}")
    logging.info(f"finished transcription of {clip_name} - {get_timestamp()}")
    return transc_res


def postprocess_transc(
    tscript_dir,
    mdata_dir,
    merge_files=False,
    linebyline=True,
    verbose=False,
) -> None:
    """
    postprocess_transc - postprocess the transcribed text by consolidating the text and metadata

    Parameters
    ----------
    tscript_dir : str, path to the directory containing the transcribed text files
    mdata_dir : str, path to the directory containing the metadata files
    merge_files : bool, optional, by default False, if True, create a new file that contains all text and metadata merged together
    linebyline : bool, optional, by default True, if True, split the text into sentences
    verbose : bool, optional

    Returns
    -------
    str, filepath to the "complete" output directory
    """
    logging.info(
        f"Starting postprocessing of transcribed text with params {locals()}"
    )
    if verbose:
        print("Starting to postprocess transcription @ {}".format(get_timestamp()))

    if merge_files:
        digest_txt_directory(tscript_dir, iden=f"orig_transc_{get_timestamp()}")
        digest_txt_directory(
            mdata_dir,
            iden=f"meta_{get_timestamp()}",
            make_folder=True,
        )

    txt_files = find_ext_local(
        tscript_dir, req_ext=".txt", verbose=verbose, full_path=False
    )

    kw_all_vids = pd.DataFrame()

    for this_transc in tqdm(
        txt_files,
        total=len(txt_files),
        desc="Processing transcribed audio",
    ):
        # Extract keywords directly from the transcription file
        with open(os.path.join(tscript_dir, this_transc), 'r', encoding='utf-8') as f:
            text = f.read()

        qk_df = quick_keys(
            filepath=tscript_dir,
            filename=this_transc,
            num_kw=25,
            max_ngrams=3,
            save_db=False,
            verbose=verbose,
        )

        kw_all_vids = pd.concat([kw_all_vids, qk_df], axis=1)

    # save overall transcription file
    kwdb_fname = f"YAKE - all keywords for run at {get_timestamp()}.csv"
    kw_all_vids.to_csv(
        join(tscript_dir, kwdb_fname),
        index=True,
    )

    return tscript_dir


def transcribe_dir(
    input_dir: str,
    chunk_length: int = 30,
    model_id: str = "openai/whisper-base",
    language: str = None,
    move_comp=False,
    join_text=False,
    print_memory_usage=False,
    verbose=False,
    generate_srt=False,
):
    """
    transcribe_dir - transcribe all videos in a directory

    :param str input_src: the path to the directory containing the videos to transcribe
    :param int chunk_length: the length of the chunks to split the audio into, in seconds. Default is 30 seconds
    :param str model_id: the model id to use for the transcription. Default is openai/whisper-base
    :param bool move_comp: if True, move the completed files to a new folder
    :param bool join_text: if True, join all lines of text into one long string
    :param bool print_memory_usage: if True, print the memory usage of the system during the transcription
    :param bool verbose: if True, print out more information
    :param bool generate_srt: if True, generate SRT files alongside transcriptions
    :param str language: Language code (e.g. 'en') or None for auto-detect

    Returns
    -------
    :return str, str: the path to the directory of transcribed text files, and the path to the directory of metadata files
    """
    st = time.perf_counter()

    directory = os.path.abspath(input_dir)
    linebyline = not join_text
    logging.info(f"Starting transcription pipeline @ {get_timestamp(True)}" + "\n")
    print(f"\nLoading models @ {get_timestamp(True)} - may take some time...")
    print("if RT seems excessive, try --verbose flag or checking logfile")

    _is_whisper = "whisper" in model_id.lower()

    if _is_whisper:
        logging.info(f"whisper model detected, using {'auto-detect' if language is None else language} language")
        if chunk_length != 30:
            warnings.warn(
                f"you have set chunk_length to {chunk_length}, but whisper models default to 30s chunks. strange things may happen"
            )

    processor, model = (
        load_whisper_modules(model_id, language=language)
        if _is_whisper
        else load_wav2vec2_modules(model_id)
    )

    approved_files = []
    for ext in get_av_fmts():  # load vid2cleantxt inputs
        approved_files.extend(find_ext_local(directory, req_ext=ext, full_path=False))

    print(f"\nFound {len(approved_files)} audio or video files in {directory}")

    storage_locs = setup_out_dirs(directory)  # create and get output folders
    for filename in tqdm(
        approved_files,
        total=len(approved_files),
        desc="transcribing...",
    ):
        t_results = (
            transcribe_video_whisper(
                model=model,
                processor=processor,
                clip_directory=directory,
                clip_name=filename,
                chunk_dur=chunk_length,
                print_memory_usage=print_memory_usage,
                verbose=verbose,
                generate_srt=generate_srt,
                language=language,
            )
            if _is_whisper
            else transcribe_video_wav2vec(
                model=model,
                processor=processor,
                clip_directory=directory,
                clip_name=filename,
                chunk_dur=chunk_length,
                print_memory_usage=print_memory_usage,
                verbose=verbose,
            )
        )

        if move_comp:
            move2completed(directory, filename=filename)  # move src to completed folder

    # postprocess the transcriptions
    out_p_tscript = storage_locs["t_out"]
    out_p_metadata = storage_locs["m_out"]
    processed_dir = postprocess_transc(
        tscript_dir=out_p_tscript,
        mdata_dir=out_p_metadata,
        merge_files=False,
        verbose=verbose,
        linebyline=linebyline,
    )

    logging.info(f"Finished transcription pipeline @ {get_timestamp(True)}" + "\n")
    logging.info(f"Total time: {round((time.perf_counter() - st)/60, 3)} mins")

    return processed_dir, out_p_metadata


def get_parser():
    """
    get_parser - a helper function for the argparse module
    Returns: argparse.ArgumentParser object
    """

    parser = argparse.ArgumentParser(
        description="Transcribe a directory of videos using transformers"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="path to directory containing video files to be transcribed",
    )

    parser.add_argument(
        "-move",
        "--move-input-vids",
        required=False,
        default=False,
        action="store_true",
        help="if specified, will move files that finished transcription to the completed folder",
    )

    parser.add_argument(
        "-m",
        "--model",
        required=False,
        default="openai/whisper-base",
        help="huggingface ASR model name. try 'facebook/wav2vec2-base-960h' if issues running default.",
    )

    parser.add_argument(
        "-cl",
        "--chunk-length",
        required=False,
        default=30,
        type=int,
        help="Duration of .wav chunks (in seconds) that the transformer model will be fed. decrease if you run into memory issues",
    )

    parser.add_argument(
        "--join-text",
        required=False,
        default=False,
        action="store_true",
        help="Save the transcribed text as a single line of text instead of one line per sentence",
    )

    parser.add_argument(
        "--print-memory-usage",
        required=False,
        default=False,
        action="store_true",
        help="Print memory usage updates during transcription",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        required=False,
        default=False,
        action="store_true",
        help="print out more information",
    )

    parser.add_argument(
        "--generate-srt",
        action="store_true",
        default=False,
        help="Generate an SRT file alongside the transcribed text. Note: timestamps may not be perfectly synced with speech",
    )

    parser.add_argument(
        "-l",
        "--language",
        required=False,
        default=None,
        help="Language code (e.g. 'en' for English). If not specified, will auto-detect language",
    )

    return parser


# TODO: change to pathlib from os.path

if __name__ == "__main__":

    # parse the command line arguments
    args = get_parser().parse_args()
    input_src = str(args.input_dir)
    # TODO: add output directory from user arg
    move_comp = args.move_input_vids
    chunk_length = int(args.chunk_length)
    model_id = str(args.model)
    join_text = args.join_text
    print_memory_usage = args.print_memory_usage
    is_verbose = args.verbose
    generate_srt = args.generate_srt
    language = args.language

    output_text, output_metadata = transcribe_dir(
        input_dir=input_src,
        chunk_length=chunk_length,
        model_id=model_id,
        language=language,
        move_comp=move_comp,
        join_text=join_text,
        print_memory_usage=print_memory_usage,
        verbose=is_verbose,
        generate_srt=generate_srt,
    )

    print(
        f"Complete. Relevant files for run are in: \n{output_text} \n{output_metadata}"
    )

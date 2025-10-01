"""
Truth Weaver: An End-to-End Voice Deception Analysis Tool

This script provides a unified pipeline for the Whispering Shadows Mystery.
It orchestrates the transcription of raw audio files and performs a subsequent
deception analysis using advanced semantic metrics and a large language model.
"""

import os
import re
import sys
import json
import time
import torch
import librosa
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from sentence_transformers import SentenceTransformer
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from groq import Groq
except ImportError:
    print("Error: Required packages are not installed.")
    print("Please run: pip install torch transformers sentence-transformers groq librosa soundfile")
    sys.exit(1)


STT_MODEL_NAME = "openai/whisper-base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_FORMATS = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'}
API_WAIT_TIME = 2


def print_header():
    """Prints the application's title and a brief on the methodology."""
    print("\n" + "="*80)
    print(" " * 32 + "Truth Weaver" + " " * 36)
    print(" " * 24 + "An AI Deception Detection and Analysis Tool" + " " * 23)
    print("="*80)
    print("\nMethodology:")
    print("  This tool analyzes transcripts using four key semantic metrics:")
    print("\n  1. Inter-Session Consistency (k): Measures consistency across all sessions.")
    print("     Formula: k = mean((1 + cos_sim(session_i, topic_vector)) / 2)")
    print("\n  2. Intra-Session Coherence (w): Measures how focused and sequential a story is.")
    print("     Formula: w = mean((1 + cos_sim(session_i, session_{i+1})) / 2)")
    print("\n  3. Semantic Drift (d): Tracks how much the narrative's meaning changes over time.")
    print("     Formula: d = sum(euclidean_dist(session_i, session_{i+1}))")
    print("\n  4. Deception Index (D): A composite score indicating deception likelihood.")
    print("     Formula: D = ((1 + d) / (w + ε)) * (1 - product(k_i))")
    print("-" * 80)

def get_api_key():
    """Prompts the user for their Groq API key."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        api_key = input("\nPlease enter your Groq API key to proceed: ").strip()
    if not api_key:
        print("\nError: An API key is required to run the analysis.")
        return None
    return api_key

def transcribe_audio_files(directory_path, output_file="transcribed.txt"):
    """
    Transcribes all audio files in a directory using a speech-to-text model.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        print(f"\nError: Directory not found at '{directory_path}'")
        return False

    audio_files = [p for p in dir_path.rglob('*') if p.suffix.lower() in SUPPORTED_FORMATS]
    if not audio_files:
        print(f"\nNo supported audio files found in '{directory_path}'.")
        return False

    audio_files.sort()

    print(f"\nLoading transcription model '{STT_MODEL_NAME}' on {DEVICE}...")
    try:
        processor = WhisperProcessor.from_pretrained(STT_MODEL_NAME)
        model = WhisperForConditionalGeneration.from_pretrained(STT_MODEL_NAME).to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading transcription model: {e}")
        return False

    print(f"\nFound {len(audio_files)} audio files. Starting transcription...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, audio_file in enumerate(audio_files, 1):
            print(f"  [{i}/{len(audio_files)}] Processing: {audio_file.name}")
            try:
                audio, sr = librosa.load(str(audio_file), sr=16000, mono=True)
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(DEVICE)

                with torch.no_grad():
                    generated_ids = model.generate(input_features)
                
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                
                clean_text = re.sub(r'[^a-z\s]', '', transcription.lower())
                clean_text = re.sub(r'\s+', ' ', clean_text).strip()
                
                f.write(f"{audio_file.name}: {clean_text}\n")
            except Exception as e:
                print(f"    - Failed to process {audio_file.name}: {e}")

    print(f"\nTranscription phase complete. Output saved to '{output_file}'.")
    return True

def analyze_transcripts(transcript_file, api_key, output_file="PrelimsSubmission.json"):
    """
    Analyzes a transcript file for deception and generates a JSON report.
    """
    print(f"\nLoading analysis models...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        groq_client = Groq(api_key=api_key)
    except Exception as e:
        print(f"Error loading analysis models: {e}")
        return

    print("Parsing and clustering transcripts by user...")
    user_sessions = _parse_and_cluster(transcript_file)
    if not user_sessions:
        return

    all_results = []
    total_users = sum(1 for sessions in user_sessions.values() if len(sessions) >= 2)
    processed_count = 0
    
    print(f"\nStarting deception analysis for {total_users} users...")

    for shadow_id, sessions in user_sessions.items():
        if len(sessions) < 2:
            continue
        
        processed_count += 1
        print(f"\n--- [{processed_count}/{total_users}] Analyzing: {shadow_id} ---")
        
        session_texts = [s['transcript'] for s in sessions]
        metrics = _calculate_metrics(session_texts, embedding_model)
        
        (norm_k, norm_w, norm_d, norm_D, session_focus_scores) = metrics

        print("  Overall Metrics:")
        print(f"    - Inter-Session Consistency: {norm_k:+.4f}")
        print(f"    - Intra-Session Coherence:   {norm_w:+.4f}")
        print(f"    - Semantic Drift:            {norm_d:+.4f}")
        print(f"    - Deception Index:           {norm_D:+.4f}")
        print("  Individual Session Focus:")
        for i, score in enumerate(session_focus_scores):
            print(f"    - Session {i+1} Focus Score:     {score:+.4f}")

        system_prompt, user_prompt = _create_llm_prompt(shadow_id, sessions, metrics)
        
        analysis = _query_llm_with_retry(groq_client, system_prompt, user_prompt)
        
        if analysis:
            all_results.append(analysis)
        else:
            print(f"    - LLM analysis failed for {shadow_id}. A fallback entry will be created.")
            all_results.append(_create_fallback_analysis(shadow_id, metrics))

        time.sleep(API_WAIT_TIME)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
        
    print(f"\nAnalysis phase complete. Final report saved to '{output_file}'.")

def _parse_and_cluster(file_path):
    """Helper to parse the transcript file."""
    user_sessions = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match(r"^(.*?)_(\d{4})_(\d+)\.mp3:\s*(.*)$", line.strip())
                if match:
                    user, year, session_num, transcript = match.groups()
                    shadow_id = f"{user.strip()}_{year}"
                    if shadow_id not in user_sessions:
                        user_sessions[shadow_id] = []
                    user_sessions[shadow_id].append({'session': int(session_num), 'transcript': transcript.strip()})
    except FileNotFoundError:
        print(f"Error: Transcript file not found at '{file_path}'")
        return {}
    
    for shadow_id in user_sessions:
        user_sessions[shadow_id].sort(key=lambda x: x['session'])
    return user_sessions

def _calculate_metrics(texts, model):
    """Helper to calculate the four semantic metrics and per-session focus."""
    if not texts:
        return 0, 0, 0, 0, []
        
    embeddings = model.encode(texts, convert_to_tensor=True, device=DEVICE)
    topic_vector = torch.mean(embeddings, dim=0)
    
    sims = torch.nn.functional.cosine_similarity(embeddings, topic_vector.unsqueeze(0))
    session_focus_scores = [((1 + s) / 2).item() for s in sims]
    
    consistency = torch.mean((1 + sims) / 2).item()

    coherence = 0
    if len(embeddings) > 1:
        coherence_sims = [torch.nn.functional.cosine_similarity(embeddings[i], embeddings[i+1], dim=0) for i in range(len(embeddings) - 1)]
        coherence = torch.mean(torch.tensor([(1 + s)/2 for s in coherence_sims])).item()

    drift = 0
    if len(embeddings) > 1:
        dists = [torch.norm(embeddings[i+1] - embeddings[i]) for i in range(len(embeddings) - 1)]
        drift = torch.sum(torch.tensor(dists)).item()

    consistency_product = np.prod(session_focus_scores)
    deception_index = ((1 + drift) / (coherence + 1e-6)) * (1 - consistency_product)
    
    norm_k = 2 * consistency - 1
    norm_w = 2 * coherence - 1
    norm_d = 2 * (drift / (1 + drift)) - 1 if drift > 0 else -1.0
    norm_D = 2 * (deception_index / (1 + abs(deception_index))) - 1
    
    return norm_k, norm_w, norm_d, norm_D, session_focus_scores

def _create_llm_prompt(shadow_id, sessions, metrics):
    """Helper to generate the LLM prompt with metrics and a one-shot example."""
    norm_k, norm_w, norm_d, norm_D, session_focus = metrics
    transcripts = "\n".join([f"  Session {s['session']}: \"{s['transcript']}\"" for s in sessions])
    session_focus_str = "\n".join([f"  - Session {i+1} Focus: {score:+.4f}" for i, score in enumerate(session_focus)])

    system_prompt = "You are an expert AI detective. Your mission is to analyze interview transcripts to uncover the truth about a candidate's skills. You must synthesize all information and respond *only* with a single, valid JSON object, with no other text, comments, or formatting."

    user_prompt = f"""
**Objective:** Analyze the provided case file to determine the subject's true skills and identify any deceptive claims.

**Example Case & Required Output Format:**
- **Case File:** phoenix_2024
- **Transcripts:**
  Session 1: "I’ve mastered Python for 6 years... built incredible systems..."
  Session 2: "Actually... maybe 3 years? Still learning advanced..."
  Session 3: "LED A TEAM OF FIVE! EIGHT MONTHS! MACHINE LEARNING!"
  Session 4: "I... I work alone mostly... never been comfortable with... with people..."
  Session 5: "*sobbing* Just 2 months debugging... I’m not... I’m not what they think..."
- **Required JSON Output:**
```json
{{
  "shadow_id": "phoenix_2024",
  "revealed_truth": {{
    "programming_experience": "3-4 years",
    "programming_language": "python",
    "skill_mastery": "intermediate",
    "leadership_claims": "fabricated",
    "team_experience": "individual contributor",
    "skills and other keywords": ["Machine Learning"]
  }},
  "deception_patterns": [
    {{
      "lie_type": "experience_inflation",
      "contradictory_claims": ["6 years", "3 years"]
    }}
  ]
}}
```


Generate in strict json format i.e. {{"key":[1,2,]}} is not valid, only {{"key":[1,2]}} is valid so do not put , delimiter at the end of lists or dicts.
---

**Current Case File for Analysis:**

**Case ID:** {shadow_id}

**Transcripts:**
{transcripts}

**Semantic Analysis (Scores from -1 to 1):**
- Overall Consistency: {norm_k:+.4f}
- Overall Coherence:   {norm_w:+.4f}
- Overall Drift:       {norm_d:+.4f}
- Deception Index:     {norm_D:+.4f}

**Individual Session Focus:**
{session_focus_str}

**Instructions:**
Based on all the evidence (transcripts and metrics), generate a single JSON object for case **{shadow_id}** that reveals the most likely truth and identifies deception patterns. Your response must strictly adhere to the schema shown in the example. Crucially, the `contradictory_claims` array should contain only the conflicting phrases themselves, without mentioning session numbers (e.g., use "6 years", not "6 years (Session 1)").
"""
    return system_prompt, user_prompt

def _query_llm_with_retry(client, system, user):
    """Helper to query the LLM with retry logic."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=1024,
                temperature=0.0
            )
            text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            print(f"    - LLM query attempt {attempt + 1} failed: {e}")
            time.sleep(API_WAIT_TIME)
    return None

def _create_fallback_analysis(shadow_id, metrics):
    """Helper to create a fallback JSON when the LLM fails."""
    _, _, _, norm_D, _ = metrics
    deception_patterns = []
    if norm_D > 0.5:
        deception_patterns.append({
            "lie_type": "inconsistent_claims",
            "contradictory_claims": ["High semantic deception index detected", "LLM analysis failed"]
        })
    return {
        "shadow_id": shadow_id,
        "revealed_truth": {
            "programming_experience": "unknown", "programming_language": "unknown",
            "skill_mastery": "unknown", "leadership_claims": "unknown",
            "team_experience": "unknown", "skills and other keywords": []
        },
        "deception_patterns": deception_patterns
    }

def main():
    """Main application workflow."""
    print_header()
    api_key = get_api_key()
    if not api_key:
        return

    audio_dir = input("Enter the path to the audio directory: ").strip().strip('"')
    
    if transcribe_audio_files(audio_dir):
        analyze_transcripts("transcribed.txt", api_key)

    print("\nTruth Weaver analysis complete.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

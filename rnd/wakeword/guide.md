# Wake Word Recording Guide

Step-by-step guide for creating training data for a custom wake word model.

## Audio Requirements

All audio must match these specs (scripts handle this automatically):

| Parameter   | Value          |
|-------------|----------------|
| Sample rate | 16,000 Hz      |
| Channels    | 1 (mono)       |
| Format      | WAV (PCM 16-bit)|
| Duration    | 0.5–3.0 seconds |

## What You Need

- A quiet room (background noise under 40 dB — no TV, music, fans if possible)
- A microphone (built-in laptop mic, USB mic, or headset)
- For external recordings: any phone, tablet, or laptop with a voice recorder app

## Recording Positive Samples (the wake word)

These are recordings of the actual wake phrase (e.g., "hey argus").

### How Many

| Source                | Minimum | Recommended |
|-----------------------|---------|-------------|
| Per speaker (mic)     | 20      | 50          |
| Per speaker (phone)   | 10      | 20          |
| Speakers total        | 2       | 4+          |

More speakers = better generalization. Ask family members, friends, colleagues.

### How to Record

1. Run `dataset/record_positive.py` — it will guide you through each recording
2. Speak naturally, as if calling out to someone in the room
3. Vary your delivery across recordings:

| Variation          | Examples                                        |
|--------------------|-------------------------------------------------|
| **Volume**         | Normal, slightly louder, slightly softer         |
| **Speed**          | Normal pace, slightly faster, slightly slower    |
| **Distance**       | 0.5m, 1m, 2m, 3m from mic                       |
| **Tone**           | Neutral, questioning, casual, calling out        |
| **Position**       | Facing mic, turned slightly left/right           |
| **Room**           | Different rooms if possible                      |

4. Do NOT whisper or shout — stay within your normal speaking range
5. Leave a brief natural pause before and after the phrase (the script handles trimming)

### Common Mistakes to Avoid

- **Too consistent**: saying it exactly the same way every time
- **Too fast**: rushing through recordings — take 1.5s pause between each
- **Background noise**: TV, music, or conversations in the background
- **Too close to mic**: causes clipping/distortion (stay 30cm+ away)
- **Reading voice**: speak naturally, not like reading from a script

## Recording Negative Samples

Negative samples teach the model what is NOT the wake word.

### Types of Negatives

| Type                | What to record                                  | Duration    |
|---------------------|-------------------------------------------------|-------------|
| **Ambient**         | Room silence, fridge hum, AC, outdoor noise      | 2–5 min     |
| **Conversation**    | Normal speech NOT containing the wake phrase     | 2–5 min     |
| **Similar phrases** | Phonetically similar words (see below)           | 20–50 clips |
| **Media audio**     | TV, music, podcasts playing in the room          | 2–5 min     |

### Adversarial Negatives (Important!)

Record these phrases that sound similar to your wake word.
The model needs to learn to reject them.

For "hey argus":
- "hey marcus", "hey august", "hey are us"
- "hey gorgeous", "the argus", "hey artists"
- "hey", "argus" (each word alone)

For any custom phrase, think of:
- Rhyming words and near-homophones
- Substrings of the phrase
- Words with similar vowel patterns

Record 5–10 clips of each adversarial phrase, per speaker.

## Recording from External Devices

Recordings from phones, tablets, and other laptops add valuable microphone
diversity to your dataset.

### Recommended Apps

| Platform | App                                      |
|----------|------------------------------------------|
| Android  | Built-in Voice Recorder, Easy Voice Recorder |
| iPhone   | Built-in Voice Memos                     |
| Laptop   | Audacity (free), built-in Sound Recorder |

### Settings on External Devices

Configure your recording app to:
- **Format**: WAV or M4A (WAV preferred, M4A will be converted)
- **Sample rate**: 16 kHz or higher (will be resampled)
- **Quality**: Standard/Normal (not ultra-low bitrate)

If you can't configure the format, any format works — `import_external.py`
handles conversion.

### How to Transfer

1. Record on the device
2. Transfer files to your computer (USB, AirDrop, email, cloud)
3. Place files in `data/external/<speaker_name>/`
4. Run `dataset/import_external.py` — it normalizes everything

### File Naming Convention

When importing external files, organize by speaker:

```
data/external/
  alex/
    hey_argus_01.wav
    hey_argus_02.wav
    ...
  maria/
    hey_argus_01.m4a
    hey_argus_02.m4a
    ...
```

The import script auto-detects speaker names from folder names.

## Dataset Structure After Collection

After running all dataset scripts, you'll have:

```
data/
  positive/              # real mic recordings (wake word)
    alex/
      positive_001.wav
      positive_002.wav
    maria/
      positive_001.wav
  negative/              # real mic recordings (not wake word)
    ambient_001.wav
    conversation_001.wav
    adversarial_hey_marcus_001.wav
  synthetic/             # Piper TTS generated
    synthetic_speaker042_001.wav
    synthetic_speaker042_002.wav
  external/              # imported from other devices (normalized)
    alex/
      hey_argus_01.wav
    maria/
      hey_argus_01.wav
  augmented/             # augmented copies of all positives
    aug_positive_001_noise.wav
    aug_positive_001_rir.wav
  noise/                 # downloaded background noise
  rir/                   # downloaded room impulse responses
```

## Quality Checklist

Before moving to training, verify:

- [ ] At least 20 real positive recordings per speaker
- [ ] At least 2 different speakers
- [ ] At least 10 adversarial negative clips per speaker
- [ ] At least 5 minutes of ambient negative audio
- [ ] Synthetic samples generated (10,000+)
- [ ] All files validated with `validate_dataset.py`
- [ ] No clipped audio (peak > -1 dBFS)
- [ ] No silent files (peak < -40 dBFS)
- [ ] All files are 16 kHz mono WAV

Run `dataset/validate_dataset.py` to check all of the above automatically.

## Tips for Best Results

1. **Diversity matters more than quantity** — 50 varied recordings beat 200 identical ones
2. **Real recordings are 3x more valuable** than synthetic in training
3. **Test in deployment conditions** — if Argus will be in the kitchen, record some samples in the kitchen
4. **Include children/elderly** if they'll use the system — voices differ significantly by age
5. **Re-record if accuracy is poor** — more data from failing conditions helps most
6. **The wake phrase matters** — 3+ syllables with distinct consonants work best

# System Architecture

## Multi-Factor Recognition System

### Overview

A modular 2-factor (expandable to 3-factor) authentication system combining PIN codes and face recognition to secure access to protected media folders.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Multi-Factor Recognition System                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│  Main App    │          │  Dashboard   │          │  Enrollment  │
│  (main.py)   │          │  (Flask)     │          │    Tool      │
└──────────────┘          └──────────────┘          └──────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │   Authentication Orchestrator │
                    │   (authenticator_2factor.py)  │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│ PIN Module   │          │ Face Module  │          │Voice Module  │
│              │          │              │          │  (Optional)  │
│ • Verify PIN │          │ • Load       │          │ • Speaker    │
│ • Hash PIN   │          │   Encodings  │          │   Verify     │
│ • Validate   │          │ • Camera     │          │ • Phrase     │
│              │          │   Capture    │          │   Match      │
└──────────────┘          │ • Face Match │          └──────────────┘
                          └──────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│   AI Model   │          │  AI Model    │          │  AI Model    │
│              │          │              │          │              │
│   bcrypt     │          │ face_recog   │          │ SpeechBrain  │
│   hashing    │          │ (dlib-based) │          │ ECAPA-TDNN   │
└──────────────┘          └──────────────┘          └──────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      Data Management          │
                    │    (yaml_manager.py)          │
                    └───────────────┬───────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐          ┌──────────────┐
│  users.yaml  │          │auth_logs.yaml│          │  Encodings   │
│              │          │              │          │              │
│ • User data  │          │ • Auth logs  │          │ • Face .npy  │
│ • PINs       │          │ • Statistics │          │ • Voice .npy │
│ • Paths      │          │ • Timestamps │          │              │
└──────────────┘          └──────────────┘          └──────────────┘
```

## Authentication Flow

```
User Access Request
        │
        ▼
┌──────────────────┐
│  Enter Username  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Enter PIN      │────────► PIN Verification ────┐
└────────┬─────────┘                               │
         │                                         │
         ▼                                         ▼
┌──────────────────┐                         ┌─────────┐
│ Camera Capture   │────────► Face Match ────│  FAIL   │
└────────┬─────────┘                         └─────────┘
         │                                         ▲
         ▼                                         │
┌──────────────────┐                              │
│ (Voice Record)   │────────► Voice Match ────────┘
└────────┬─────────┘          (Optional)
         │
         ▼
┌──────────────────┐
│  All Verified?   │
└────────┬─────────┘
         │
         ▼
    ┌────┴────┐
    │  YES    │
    └────┬────┘
         │
         ▼
┌──────────────────┐
│  Grant Access    │
│  to Protected    │
│  Folder          │
└──────────────────┘
         │
         ▼
┌──────────────────┐
│  Log Attempt     │
│  Show Contents   │
└──────────────────┘
```

## Module Architecture

### Core Modules

**1. PIN Module (`auth/pin_module.py`)**
- PIN format validation
- bcrypt password hashing
- PIN verification against stored hash
- Support for plain text (legacy) and hashed PINs

**2. Face Module (`auth/face_module.py`)**
- Load/save face encodings
- Generate encodings from images
- Real-time camera capture
- Face matching with configurable tolerance
- Uses dlib HOG detector and face_recognition library

**3. Voice Module (`auth/voice_module.py`)** - Optional
- Audio recording via PyAudio
- Speaker embedding generation (SpeechBrain)
- Phrase verification (speech-to-text)
- Cosine similarity comparison

### Data Layer

**YAML Manager (`database/yaml_manager.py`)**
- User CRUD operations
- Authentication logging
- Statistics generation
- File-based storage for simplicity

**Data Files:**
- `users.yaml`: User profiles, PINs, file paths, settings
- `auth_logs.yaml`: Timestamped authentication attempts
- `*.npy`: NumPy arrays for face/voice encodings

### Application Layer

**Main App (`main.py`)**
- Interactive CLI interface
- User authentication workflow
- Statistics viewing
- Enrollment status checking

**Dashboard (`dashboard/app.py`)**
- Flask web server
- Real-time statistics
- User management UI
- Authentication logs viewer
- REST API endpoints

**Enrollment (`enrollment/enroll_user.py`)**
- Interactive enrollment process
- Face image processing
- Voice sample collection
- User data initialization

## Data Storage

### File Structure

```
data/
├── users.yaml              # User profiles
├── auth_logs.yaml          # Authentication logs
├── faces/                  # Training images
│   ├── Ibrahim/
│   └── Ahmed/
├── face_encodings/         # Computed encodings
│   ├── Ibrahim.npy
│   └── Ahmed.npy
├── voices/                 # Voice samples (optional)
│   ├── Ibrahim/
│   └── Ahmed/
└── voice_embeddings/       # Voice embeddings (optional)
    ├── Ibrahim.npy
    └── Ahmed.npy

protected_media/            # Protected content
├── Ibrahim/
└── Ahmed/
```

### users.yaml Schema

```yaml
<username>:
  pin: "<hashed_pin_or_plain>"
  face_encodings_path: "data/face_encodings/<username>.npy"
  voice_embedding_path: "data/voice_embeddings/<username>.npy"  # optional
  assigned_folder: "protected_media/<username>"
  is_active: true/false
```

### auth_logs.yaml Schema

```yaml
- timestamp: "YYYY-MM-DD HH:MM:SS"
  username: "<username>"
  pin_success: true/false
  face_success: true/false
  voice_success: true/false
  overall_success: true/false
  failure_reason: "<reason>" or null
```

## AI Models

### Face Recognition
- **Library**: face_recognition (dlib wrapper)
- **Algorithm**: 
  - Face detection: HOG (Histogram of Oriented Gradients)
  - Face encoding: 128-dimension embedding
  - Matching: Euclidean distance with threshold
- **Accuracy**: Configurable via `FACE_TOLERANCE` (default: 0.5)

### Voice Recognition (Optional)
- **Speaker Verification**: 
  - Model: SpeechBrain ECAPA-TDNN
  - Pre-trained on VoxCeleb dataset
  - Embedding size: 192 dimensions
  - Matching: Cosine similarity
- **Phrase Verification**:
  - Google Speech API or Vosk (offline)
  - Expected phrase: "My voice is my passport verify me"

## Security Features

1. **PIN Security**
   - bcrypt hashing (configurable)
   - Salted password storage
   - Fallback to plain text for legacy support

2. **Biometric Security**
   - Multi-encoding storage (multiple face samples)
   - Configurable matching thresholds
   - Time-limited authentication windows

3. **Access Control**
   - User-specific protected folders
   - No access without full authentication
   - Comprehensive audit logging

4. **Privacy**
   - Local data storage only
   - No cloud dependencies (except optional speech API)
   - Offline operation capability

## Configuration

All settings in `config.py`:

```python
# Face recognition
FACE_TOLERANCE = 0.5          # Match threshold
FRAME_DOWNSCALE = 0.25        # Processing speed
PROCESS_EVERY_N = 2           # Frame skip rate

# Voice recognition
VOICE_PHRASE = "..."          # Required phrase
VOICE_SIMILARITY_THRESHOLD = 0.75
AUDIO_SAMPLE_RATE = 16000

# Security
ENABLE_PIN_HASHING = True     # Use bcrypt
PIN_LENGTH = 4                # PIN digits

# Dashboard
DASHBOARD_PORT = 5000
DASHBOARD_DEBUG = True
```

## Scalability

### Current: File-Based (YAML)
- ✓ Simple deployment
- ✓ No database setup required
- ✓ Easy backup (just copy files)
- ✓ Suitable for: 1-10 users

### Future: Database Migration
- For 10+ users, migrate to SQLite/PostgreSQL
- Replace `yaml_manager.py` with SQL ORM
- No changes to auth modules needed
- Maintain same API interface

## Extension Points

1. **Additional Auth Factors**
   - Add new module in `auth/`
   - Implement verification method
   - Update orchestrator

2. **Different Storage**
   - Implement new manager (e.g., `sql_manager.py`)
   - Keep same interface as `yaml_manager.py`
   - Update `config.py`

3. **Custom AI Models**
   - Replace face_recognition with custom CNN
   - Use different voice model
   - Update respective modules

4. **Web Interface**
   - Already has REST API in dashboard
   - Can build React/Vue frontend
   - API endpoints documented in dashboard/app.py

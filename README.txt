### Python Packages Installation:
1) `cd` to the root directory of this project
2) run `pip install -r requirements.txt`

### External Dependencies to install (Important)
##### **1) FFmpeg**
1. Go to [https://www.gyan.dev/ffmpeg/builds/)
2. Go to "Release builds"
3. Download the **release full build** `.zip` file.
4. Extract it, e.g., to `C:\ffmpeg`.

**Add FFmpeg to System PATH**
1. Open "Edit the system environment variables" (search in Start menu).
2. Click **Environment Variables**.
3. In "System variables", find `Path`, click Edit.
4. Click New, add the path to the `bin` folder in your FFmpeg install, e.g.:

   ```
   C:\ffmpeg\bin
   ```
5. Click OK, OK, OK.

**Verify Installation**

* Open a **new** Command Prompt (important! Restart it so PATH is updated)
* Run:

  ```
  ffmpeg -version
  ```
---


### Windows Applications to install
These applications are for live demonstration of the project by letting you create your own deepfakes.
1) VoiceAI
  https://voice.ai/
2) OBS
  https://obsproject.com/download
3) DeepFace Live
  https://mega.nz/folder/m10iELBK#Y0H6BflF9C4k_clYofC7yA/file/bs9FlKrS

To start the project:
1) run python app.py 
2) start DeepFace Live, and turn on all the settings available
3) start OBS and start virtual camera

### Python Setup
1) Ensure python version 3.10
2) Recommended to use a virtual environment in Python or Anaconda

### Python Packages Installation:
1) `cd` to the root directory of this project
2) run `pip install -r requirements.txt`

### External Dependencies to install (Important)
#### **1) FFmpeg**
1. Go to [https://www.gyan.dev/ffmpeg/builds/]
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


#### **2) Microsoft Visual C++ Redistributable**
1. Download and setup using this link: [https://aka.ms/vs/17/release/vc_redist.x86.exe]

#### **3) Microsoft Visual C++ Build Tools >= v14.0**
1. Download the installer using this link: [https://visualstudio.microsoft.com/visual-cpp-build-tools/]
2. Run the installer and select the following option (keep default sub-options):
   
   <img width="1280" height="639" alt="image" src="https://github.com/user-attachments/assets/2d483876-87d9-44a5-9683-7893cd0bde3c" />
4. Install and exit after complete

---

### Windows Applications to install
These applications are for live demonstration of the project by letting you create your own deepfakes.
1) VoiceAI
  [https://voice.ai/]
2) OBS
  [https://obsproject.com/download]
3) DeepFace Live
  [https://mega.nz/folder/m10iELBK#Y0H6BflF9C4k_clYofC7yA/file/bs9FlKrS]

---

### To start the project:
1) `cd` to the root directory of this project
2) Ensure you are using your intended virtual environemnt / python interpreter
3) run the command `python app.py`

**If require live deepfake generation:**
4) start DeepFace Live, and turn on all the settings available
5) start OBS and start virtual camera

---

### Loading Browser Extension
1) Go to `chrome://extensions` or `edge://extensions`
2) Enable Developer Mode
3) Click on "Load unpacked"
4) Browse to the project's "browser extension" folder and select the folder

---

### Rough guide for setting up web server (for own reference):
1. Download and install nginx [https://nginx.org/download/nginx-1.29.0.zip]
2. Ensure port forwarding & DDNS is setup on router
3. Configure nginx.conf with preset config in this repo
4. Change the app.py to use Waitress's serve() instead of app.run()
5. Open a CMD to nginx installation folder and type `start nginx`
6. To restart `nginx -s reload`
7. To stop `nginx -s quit`

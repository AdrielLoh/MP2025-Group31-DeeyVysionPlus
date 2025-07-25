### Python Setup
1) Ensure python version 3.10
2) Recommended to use a virtual environment in Python or Anaconda

### Python Packages Installation:
1) `cd` to the root directory of this project
2) run `pip install -r requirements.txt`

### External Dependencies to install (Important)
#### **1) FFmpeg**
1. Go to https://www.gyan.dev/ffmpeg/builds/
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
1. Download and setup using this link: https://aka.ms/vs/17/release/vc_redist.x86.exe

#### **3) Microsoft Visual C++ Build Tools >= v14.0**
1. Download the installer using this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Run the installer and select the following option (keep default sub-options):
   
   <img width="1280" height="639" alt="image" src="https://github.com/user-attachments/assets/2d483876-87d9-44a5-9683-7893cd0bde3c" />
4. Install and exit after complete

---

### Windows Applications to install
These applications are for live demonstration of the project by letting you create your own deepfakes.
1) VoiceAI
  https://voice.ai/
2) OBS
  https://obsproject.com/download
3) DeepFace Live
  https://mega.nz/folder/m10iELBK#Y0H6BflF9C4k_clYofC7yA/file/bs9FlKrS

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

### Rough guide for setting up web server:
Setting up this project to be hosted requires some knowledge on how to setup and use Dynamic DNS, port forwarding, SSL certificates, and firewall rules. This is *optional*, and only used for live demonstrations for audiences to have a hands-on with the website.

**Downloading necessary stuff** 
1. Download and install nginx https://nginx.org/download/nginx-1.29.0.zip
2. Sign up and obtain a free Dynamic DNS hostname from NoIP https://www.noip.com (you can use any other DDNS provider too)
3. For compatibility with JS webcam APIs, get a free SSL certificate from Let's Encrypt or subscribe $3 per month to NoIP to setup HTTPS. Search Google or ChatGPT on how to generate a CSR, submit it, and obtain the SSL cert and key.
4. If usig NoIP, download and setup their Dynamic Update Client on the PC you are hosting the website on.

**Networking stuff**

5. Ensure port forwarding to 80 and 443, and DDNS is setup on router or PC.
6. Make sure to allow inbound ports 80 and 443 in your firewall.

**Nginx configuration**

7. Configure your `nginx.conf` with preset config in this repo.
8. In the `nginx.conf`, edit the `server_name` option to use your NoIP hostname (e.g., `deepvysion.ddns.net`).

**Running the server**

9. Change the `app.py` to use Waitress's `serve()` instead of `app.run()`.
10. Open a CMD to nginx installation folder and type `start nginx`.
11. To restart `nginx -s reload`.
12. To stop `nginx -s quit`.

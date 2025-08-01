### Python Setup
1) Ensure python version 3.10
2) Recommended to use a virtual environment in Python or Anaconda

---

### Python Packages Installation:
1) `cd` to the root directory of this project
2) run `pip install -r requirements.txt --no-deps`
3) Manually resolve dependency conflicts if any

---

### External Dependencies to install (Important)
#### **1) FFmpeg (Windows/ Linux)**
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

**If on Ubuntu:**
1. Install FFMPEG using `sudo apt install ffmpeg`
2. Verify using `ffmpeg -version`


#### **2) Microsoft Visual C++ Redistributable (Windows Only)**
1. Download and setup using this link: https://aka.ms/vs/17/release/vc_redist.x86.exe

#### **3) Microsoft Visual C++ Build Tools >= v14.0 (Windows Only)**
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
1) `cd` to the root directory of this project (compulsory)
2) Ensure you are using your intended virtual environemnt / python interpreter
3) run the command `python app.py` while your current working directory is the root directory of this project

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
10. Open a CMD to nginx installation folder and type `start nginx` OR `sudo systemctl start nginx`.
11. To restart `nginx -s reload` OR `sudo systemctl reload nginx`.
12. To stop `nginx -s quit` OR `sudo systemctl stop nginx`.

---

### If using AMD GPU and you want GPU acceleration:
**This guide was written on the 30th July 2025. Compatibility may have improved afterwards so do some research before attempting this.**

It is completely fine to run the project on CPU, but if you want to use GPU, this guide will tell you how to do it.

**No guide will be given for Nvidia cards here, as setting up the project to run using CUDA should be more straightforward than ROCm**

Check if your GPU supports ROCm 6.4.2:

- AMD Radeon RX 9070
- AMD Radeon RX 9070 XT
- AMD Radeon RX 9070 GRE
- AMD Radeon AI PRO R9700
- AMD Radeon RX 9060 XT
- AMD Radeon RX 7900 XTX
- AMD Radeon RX 7900 XT
- AMD Radeon RX 7900 GRE
- AMD Radeon PRO W7900
- AMD Radeon PRO W7900 Dual Slot
- AMD Radeon PRO W7800
- AMD Radeon PRO W7800 48GB
- AMD Radeon RX 7800 XT
- AMD Radeon PRO W7700
- AMD Radeon RX 7700 XT

#### Dual-boot Windows + Ubuntu 22.04.5 LTS:
ROCm does NOT provide full acceleration features to WSL2. While TensorFlow will mostly be able to work on ROCm on WSL2 (a little buggy), PyTorch will not work on GPU at all. Hence, to fully utilize GPU acceleration for both TensorFlow and PyTorch, dual-booting Ubuntu is necessary.
1. Pre-requisites: You will need a USB drive (8GB+), Rufus, and an Ubuntu ISO.
2. Go to disk manager in Windows and shrink your partition volume by the amount of storage you want to give to the Ubuntu OS (e.g. 120000 MB).
3. Download the Ubuntu 22.04.5 LTS ISO from the official website, and download Rufus too.
4. Flash your USB drive with the ISO, using ISO mode. Boot option must be UEFI non-CSM, and partition table is set to GPT. File system is FAT32 with default sector size.
5. Restart and boot into the USB drive, follow the setup guide and install Ubuntu alongside Windows Boot Manager.

#### Installing ROCm and AMD GPU drivers:
1. Go to https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/install-radeon.html
2. Follow the instructions exactly as it shows inside. It will require you to reboot the system a few times.

#### Setting up the python environment:
1. You will need to ensure python version is 3.10.x, and you have `python3-venv` package installed.
2. Run `python3 -m venv /path/to/your/env` to create a new virtual environment.
3. To activate the environment, run `source /path/to/env/bin/activate`
4. Ensure your current working directory is the root directory of this project.
4. Install all packages using `pip install -r requirements_rocm.txt --no-deps`

#### Setting up OpenPifPaf:
Since the pip version of openpifpaf was built using an old version of PyTorch, we cannot use it on the ROCm version of PyTorch. So, we need to build openpifpaf manually.

1. clone the openpifpaf repository using:
``` bash
git clone https://github.com/openpifpaf/openpifpaf.git
cd openpifpaf
```
2. Clean any previous builds:
``` bash
python setup.py clean
```
3. In `pyproject.toml` and `setup.py`, find any strings related to:
`torch==1.13.1` or `torchvision==0.14.1`
4. Change those strings to remove the specific version number (e.g. `torch==1.13.1` change to `torch`)
5. Finally, run to build `pip install --no-build-isolation --no-binary :all: .
`
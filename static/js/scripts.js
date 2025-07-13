document.addEventListener("DOMContentLoaded", function() {
    console.log("Script loaded")
    // Load the navbar
    fetch('navbar')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(data => {
            document.getElementById('navbar-placeholder').innerHTML = data;
            setupNavbarJS(); // <--- Call the navbar setup after DOM is inserted
        })
        .catch(error => {
            console.error('Error loading navbar:', error);
        });

    // Load the footer
    fetch('footer')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(data => {
            document.getElementById('footer-placeholder').innerHTML = data;
        })
        .catch(error => {
            console.error('Error loading footer:', error);
        });

    function setupNavLinks() {
        const links = document.querySelectorAll("nav ul li a");
        
        links.forEach(link => {
            link.addEventListener("click", function(e) {
                const href = this.getAttribute("href");
                if (href.startsWith("#")) {
                    e.preventDefault();
                    const targetId = href.substring(1);
                    const targetSection = document.getElementById(targetId);
                    
                    if (targetSection) {
                        window.scrollTo({
                            top: targetSection.offsetTop,
                            behavior: "smooth"
                        });
                    }
                }
            });
        });
    }

    // Put this in scripts.js, NOT inside navbar.html
    function setupNavbarJS() {
        const menuBtn = document.getElementById("mobileMenuBtn");
        const mobileMenu = document.getElementById("mobileMenu");
        const navLinks = document.querySelectorAll(".mobile-nav-link");

        if (!menuBtn || !mobileMenu) return;

        menuBtn.addEventListener("click", function () {
            mobileMenu.classList.toggle("active");
            menuBtn.classList.toggle("active");
            document.body.style.overflow = mobileMenu.classList.contains("active") ? "hidden" : "";
        });

        navLinks.forEach(link => {
            link.addEventListener("click", function () {
                mobileMenu.classList.remove("active");
                menuBtn.classList.remove("active");
                document.body.style.overflow = "";
            });
        });

        document.addEventListener("click", function (e) {
            if (
                mobileMenu.classList.contains("active") &&
                !mobileMenu.contains(e.target) &&
                !menuBtn.contains(e.target)
            ) {
                mobileMenu.classList.remove("active");
                menuBtn.classList.remove("active");
                document.body.style.overflow = "";
            }
        });
    }

    // Contact form submission
    const form = document.getElementById("contact-form");
    if (form) {
        form.addEventListener("submit", function(e) {
            e.preventDefault();
            alert("Form submitted successfully!");
        });
    }

    // Statistics count up animation
    function countUp(element, endValue, isPercentage = false) {
        let startValue = 0;
        const duration = 2500;
        const increment = endValue / (duration / 50);

        function updateCounter() {
            startValue += increment;
            if (startValue < endValue) {
                element.textContent = Math.ceil(startValue);
                if (isPercentage) {
                    element.textContent += '%';
                }
                requestAnimationFrame(updateCounter);
            } else {
                element.textContent = endValue;
                if (isPercentage) {
                    element.textContent += '%';
                }
            }
        }

        updateCounter();
    }

    function startCounting() {
        const stat1 = document.getElementById('stat1');
        const stat2 = document.getElementById('stat2');
        const stat3 = document.getElementById('stat3');
        
        if (stat1) countUp(stat1, 95820);
        if (stat2) countUp(stat2, 85, true);
        if (stat3) countUp(stat3, 96, true);
    }

    // Intersection Observer for scroll animations
    const observer = new IntersectionObserver(entries => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('visible');
                    if (entry.target.classList.contains('stat')) {
                        startCounting();
                    }
                    observer.unobserve(entry.target);
                }, index * 200); // Delay each box by 200ms
            }
        });
    }, { threshold: 0.1 });

    const statElements = document.querySelectorAll('.stat');
    statElements.forEach(stat => {
        observer.observe(stat);
    });

    const boxElements = document.querySelectorAll('.box');
    boxElements.forEach(box => {
        observer.observe(box);
    });

    const sectionElements = document.querySelectorAll('.section, .wide-section');
    sectionElements.forEach(section => {
        observer.observe(section);
    });

    // Search function for educational resources
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        const boxes = document.querySelectorAll('.box');
        searchInput.addEventListener('input', function() {
            const searchQuery = searchInput.value.toLowerCase();
            boxes.forEach(box => {
                const title = box.querySelector('h2').textContent.toLowerCase();
                const description = box.querySelector('p').textContent.toLowerCase();
                if (title.includes(searchQuery) || description.includes(searchQuery)) {
                    box.style.display = 'flex';
                } else {
                    box.style.display = 'none';
                }
            });
        });
    }

    // Drag and drop for audio upload
    const dropbox = document.getElementById('dropbox');
    const fileInput = document.getElementById('file');
    const audioPreview = document.getElementById('audio-preview');
    const uploadForm = document.getElementById('upload-form');

    if (dropbox && fileInput && audioPreview && uploadForm) {
        // dropbox.addEventListener('click', () => fileInput.click());

        dropbox.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropbox.classList.add('dragover');
        });

        dropbox.addEventListener('dragleave', () => dropbox.classList.remove('dragover'));

        dropbox.addEventListener('drop', (e) => {
            e.preventDefault();
            dropbox.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            handleFile(file);
        });

        fileInput.addEventListener('change', () => {
            const file = fileInput.files[0];
            handleFile(file);
        });

        function handleFile(file) {
            if (file) {
                const url = URL.createObjectURL(file);
                audioPreview.src = url;
                audioPreview.style.display = 'block';
                dropbox.querySelector('p').style.display = 'none';
            } else {
                audioPreview.style.display = 'none';
                dropbox.querySelector('p').style.display = 'block';
            }
        }

        uploadForm.addEventListener('submit', function(e) {
            if (!fileInput.files.length) {
                e.preventDefault();
                alert('Please select an audio file before submitting.');
                fileInput.focus();
            }
        });
    }    
});

function deleteFilesAndGoBack() {
    window.history.back();   
}

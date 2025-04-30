document.addEventListener("DOMContentLoaded", function() {
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
            setupNavLinks();
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
        const duration = 10000;
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
        
        if (stat1) countUp(stat1, 3000);
        if (stat2) countUp(stat2, 85, true);
        if (stat3) countUp(stat3, 60, true);
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
        dropbox.addEventListener('click', () => fileInput.click());

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

    // Real-Time Audio Recording
    let mediaRecorder;
    let audioChunks = [];

    const realTimeButton = document.getElementById('real-time-button');
    if (realTimeButton) {
        realTimeButton.addEventListener('click', openRecordingModal);
    }

    function openRecordingModal() {
        const modal = document.getElementById("recording-modal");
        if (modal) {
            modal.style.display = "block";
            startRecording();
        }
    }

    function closeRecordingModal() {
        const modal = document.getElementById("recording-modal");
        if (modal) {
            modal.style.display = "none";
        }
    }

    function startRecording() {
        const countdown = document.getElementById("countdown");
        const volumeMeter = document.getElementById("volume-meter");
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream);
                mediaRecorder.start();

                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(stream);
                const analyser = audioContext.createAnalyser();
                source.connect(analyser);

                const dataArray = new Uint8Array(analyser.fftSize);
                analyser.getByteTimeDomainData(dataArray);

                const updateVolumeMeter = () => {
                    analyser.getByteTimeDomainData(dataArray);
                    const normalizedValue = Math.max(...dataArray) / 128 - 1;
                    const volumePercentage = Math.min(Math.max(normalizedValue, 0), 1) * 100;
                    volumeMeter.style.width = `${volumePercentage}%`;

                    if (volumePercentage > 66) {
                        volumeMeter.style.backgroundColor = 'red';
                    } else if (volumePercentage > 33) {
                        volumeMeter.style.backgroundColor = 'yellow';
                    } else {
                        volumeMeter.style.backgroundColor = 'green';
                    }

                    requestAnimationFrame(updateVolumeMeter);
                };

                updateVolumeMeter();

                let count = 3; // Countdown duration set to 3 seconds
                const countdownInterval = setInterval(() => {
                    countdown.textContent = count;
                    if (count === 0) {
                        clearInterval(countdownInterval);
                        mediaRecorder.stop();
                        closeRecordingModal();
                    }
                    count--;
                }, 1000);

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audio = document.getElementById('recorded-audio');
                    if (audio) {
                        audio.src = audioUrl;
                        audio.style.display = 'block';
                        uploadAudio(audioBlob);
                    }
                };
            });
    }

    function uploadAudio(audioBlob) {
        const formData = new FormData();
        formData.append('file', audioBlob, 'recorded_audio.wav');

        fetch('/start_real_time_audio_analysis', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(html => {
            document.body.innerHTML = html;
        })
        .catch(error => {
            console.error('Error:', error);
            const realTimeDetectionResult = document.getElementById('real-time-detection-result');
            if (realTimeDetectionResult) {
                realTimeDetectionResult.textContent = "Error in detection.";
            }
        });
    }

    const modalCloseButton = document.getElementById("modal-close-button");
    if (modalCloseButton) {
        modalCloseButton.addEventListener("click", closeRecordingModal);
    }
    
});

function deleteFilesAndGoBack() {
    fetch('/delete_files', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        window.history.back();
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function startQuiz() {
    document.getElementById('results-container').style.display = 'none';
    document.getElementById('quiz-container').style.display = 'block';
    score = 0;
    currentQuestionIndex = 0;

    shuffle(questions);
    currentQuestions = questions.slice(0, 5);

    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = '0%';

    displayQuestion();
}

document.addEventListener('DOMContentLoaded', (event) => {
    startQuiz();
});

const questions = [
    {
        question: "What is the primary purpose of audio analysis in deepfake detection?",
        options: [
            "To detect the background noise.",
            "To enhance the video quality.",
            "To check the resolution of the audio file.",
            "To analyze the pitch, tone, and consistency of speech for signs of manipulation."
        ],
        answer: 3,
        explanation: "Audio analysis examines characteristics of speech to identify signs of manipulation, which is common in deepfakes."
    },
    {
        question: "Which of the following is a significant challenge in audio analysis?",
        options: [
            "Analyzing pitch variations.",
            "Identifying background noise.",
            "Detecting sound effects.",
            "High-quality audio is necessary for accurate analysis."
        ],
        answer: 3,
        explanation: "High-quality audio is crucial for accurate analysis, as poor quality can hinder detection accuracy."
    },
    {
        question: "What is the first step in creating a deepfake?",
        options: [
            "Collecting data (images and videos).",
            "Editing the final video.",
            "Publishing the deepfake.",
            "Training the AI model."
        ],
        answer: 0,
        explanation: "Collecting a high-quality dataset of the target individual is the first and crucial step in creating a deepfake."
    },
    {
        question: "Which AI model is commonly used to create deepfakes?",
        options: [
            "Support Vector Machines",
            "Generative Adversarial Networks (GANs)",
            "Decision Trees",
            "Convolutional Neural Networks (CNNs)"
        ],
        answer: 1,
        explanation: "GANs are commonly used in deepfake creation as they involve a generator and discriminator working together to create realistic fakes."
    },
    {
        question: "What role does a discriminator play in GANs?",
        options: [
            "Generates fake images.",
            "Distinguishes between real and fake images.",
            "Compresses video files.",
            "Creates animations."
        ],
        answer: 1,
        explanation: "The discriminator's role is to distinguish between real and fake images, improving the generator's output over time."
    },
    {
        question: "Which of the following is a method to detect deepfakes?",
        options: [
            "Manual editing.",
            "Pixel enhancement.",
            "AI-based detection tools.",
            "Image compression."
        ],
        answer: 2,
        explanation: "AI-based detection tools analyze subtle inconsistencies in visual and audio data to detect deepfakes."
    },
    {
        question: "Why is high-quality data important in creating deepfakes?",
        options: [
            "It speeds up the video creation process.",
            "It makes the AI model easier to train.",
            "It ensures better accuracy and realism.",
            "It increases the file size."
        ],
        answer: 2,
        explanation: "High-quality data ensures that the AI model can learn and replicate the target's features more accurately."
    },
    {
        question: "How does Remote Photoplethysmography (rPPG) help in detecting deepfakes?",
        options: [
            "By analyzing heartbeats.",
            "By detecting speech patterns.",
            "By identifying eye movement.",
            "By tracking blood flow through subtle changes in skin color."
        ],
        answer: 3,
        explanation: "rPPG tracks subtle changes in skin color due to blood flow, which are hard to fake in deepfakes."
    },
    {
        question: "What is the main challenge in using rPPG for deepfake detection?",
        options: [
            "It only works on high-resolution videos.",
            "It requires active participation from the subject.",
            "It is influenced by video quality and lighting conditions.",
            "It is only applicable in specific scenarios."
        ],
        answer: 2,
        explanation: "rPPG effectiveness can be compromised by poor video quality and lighting conditions."
    },
    {
        question: "What was a significant breakthrough that led to the rise of deepfakes?",
        options: [
            "Development of Convolutional Neural Networks (CNNs).",
            "Introduction of Support Vector Machines.",
            "Advances in image compression techniques.",
            "Public release of deepfake tools on platforms like Reddit and GitHub."
        ],
        answer: 3,
        explanation: "The public release of deepfake tools made the technology widely accessible, leading to both creative uses and concerns over misuse."
    },
    {
        question: "What is a common ethical concern with deepfakes?",
        options: [
            "They improve video quality.",
            "They can be used to create misleading content.",
            "They are difficult to create.",
            "They require high computational power."
        ],
        answer: 1,
        explanation: "The ability to create misleading content and spread misinformation is a major ethical concern associated with deepfakes."
    },
    {
        question: "How do deep learning models detect deepfakes?",
        options: [
            "By manually analyzing each video frame.",
            "By compressing the video files.",
            "By enhancing the resolution of videos.",
            "By using neural networks trained on real and fake datasets to spot inconsistencies."
        ],
        answer: 3,
        explanation: "Deep learning models analyze videos for subtle inconsistencies, such as unnatural facial movements, to detect deepfakes."
    },
    {
        question: "What is a major advantage of deep learning-based detection methods?",
        options: [
            "They require no training data.",
            "They do not need computational resources.",
            "They can only detect old deepfakes.",
            "They are highly accurate when trained on large, diverse datasets."
        ],
        answer: 3,
        explanation: "Deep learning models improve accuracy with large and diverse datasets, making them effective in detecting deepfakes."
    },
    {
        question: "Why is deepfake technology considered a threat to political stability?",
        options: [
            "It can create entertainment content.",
            "It enhances video quality in political campaigns.",
            "It can be used to spread false information and manipulate public opinion.",
            "It improves communication between politicians."
        ],
        answer: 2,
        explanation: "Deepfakes can be used to create false information about political figures, influencing elections and inciting unrest."
    },
    {
        question: "How can deepfakes impact personal privacy?",
        options: [
            "By creating malicious content such as revenge porn or false incriminating videos.",
            "By creating harmless content.",
            "By improving security protocols.",
            "By enhancing video quality for personal videos."
        ],
        answer: 0,
        explanation: "Deepfakes can violate personal privacy by creating malicious content, leading to severe personal and social consequences."
    },
    {
        question: "What is the role of color correction in deepfake refinement?",
        options: [
            "To change the color of the background.",
            "To make the deepfake more visually convincing by matching the color tones.",
            "To increase the brightness of the video.",
            "To enhance the audio quality."
        ],
        answer: 1,
        explanation: "Color correction helps in matching the color tones of the face and background, making the deepfake more convincing."
    },
    {
        question: "What makes deepfakes particularly difficult to detect?",
        options: [
            "Low-quality creation.",
            "Limited data.",
            "Long training times.",
            "Advanced AI techniques and the increasing realism of the fakes."
        ],
        answer: 3,
        explanation: "The increasing realism and sophistication of AI techniques make deepfakes difficult to detect."
    },
    {
        question: "Which industry is most at risk from deepfakes?",
        options: [
            "Banking",
            "Social media",
            "Education",
            "Healthcare"
        ],
        answer: 1,
        explanation: "Social media platforms are highly vulnerable to deepfake content, which can spread rapidly and cause significant harm."
    },
    {
        question: "What is the impact of deepfakes on the entertainment industry?",
        options: [
            "They improve the sound quality of music.",
            "They enhance the resolution of old movies.",
            "They allow for the creation of realistic CGI effects and virtual actors.",
            "They are used to create fake news."
        ],
        answer: 2,
        explanation: "Deepfakes allow for the creation of realistic CGI effects and virtual actors, revolutionizing the entertainment industry."
    },
    {
        question: "How can educational institutions use deepfakes positively?",
        options: [
            "By using them to create engaging and personalized learning materials.",
            "By creating fake news for students.",
            "By replacing teachers with AI models.",
            "By improving the quality of textbooks."
        ],
        answer: 0,
        explanation: "Deepfakes can be used to create engaging and personalized learning materials, making education more interactive and accessible."
    },
    {
        question: "What is a significant concern regarding deepfakes and social trust?",
        options: [
            "They enhance social interactions.",
            "They make social media more fun.",
            "They improve communication.",
            "They can lead to a general mistrust of media and information sources."
        ],
        answer: 3,
        explanation: "Deepfakes can erode trust in media and information sources, making it harder to distinguish between real and fake content."
    },
    {
        question: "How can businesses protect themselves from deepfake-enabled scams?",
        options: [
            "By ignoring deepfakes.",
            "By developing and using sophisticated detection tools.",
            "By enhancing video quality.",
            "By reducing their online presence."
        ],
        answer: 1,
        explanation: "Businesses can protect themselves by developing and using sophisticated tools to detect deepfakes and prevent scams."
    }
];


let currentQuestions = [];
let score = 0;
let currentQuestionIndex = 0;

function checkAnswer(optionElem, selectedIndex) {
    const correctAnswer = currentQuestions[currentQuestionIndex].answer;
    const explanationElem = document.querySelector('.explanation');
    const nextButton = document.querySelector('.next-button');

    if (selectedIndex === correctAnswer) {
        optionElem.style.backgroundColor = 'green';
        explanationElem.textContent = "Correct! " + currentQuestions[currentQuestionIndex].explanation;
        score++;  // Increment score for correct answer
    } else {
        optionElem.style.backgroundColor = 'red';
        explanationElem.textContent = "Incorrect. " + currentQuestions[currentQuestionIndex].explanation;
    }

    explanationElem.style.display = 'block'; // Show the explanation
    nextButton.style.display = 'inline-block'; // Show the next button

    const options = document.querySelectorAll('.quiz-options li');
    options.forEach(option => {
        option.onclick = null; // Disable all options after selection
    });
}

function updateProgressBar() {
    const progressBar = document.getElementById('progress-bar');
    const progress = (currentQuestionIndex / currentQuestions.length) * 100;
    progressBar.style.width = progress + '%';
}

function displayQuestion() {
    if (currentQuestionIndex >= currentQuestions.length) {
        showResults();
        return;
    }

    const quizContainer = document.getElementById('quiz-container');
    quizContainer.innerHTML = '';

    const questionObj = currentQuestions[currentQuestionIndex];
    const questionElem = document.createElement('div');
    questionElem.classList.add('quiz-question');
    questionElem.innerText = `Question ${currentQuestionIndex + 1}: ${questionObj.question}`;

    const optionsList = document.createElement('ul');
    optionsList.classList.add('quiz-options');

    questionObj.options.forEach((option, index) => {
        const optionElem = document.createElement('li');
        optionElem.innerText = option;
        optionElem.onclick = () => checkAnswer(optionElem, index);
        optionsList.appendChild(optionElem);
    });

    quizContainer.appendChild(questionElem);
    quizContainer.appendChild(optionsList);

    // Add a placeholder for the explanation
    const explanationElem = document.createElement('div');
    explanationElem.classList.add('explanation');
    explanationElem.style.display = 'none'; // Hide initially
    quizContainer.appendChild(explanationElem);

    // Add a button to move to the next question
    const nextButton = document.createElement('button');
    nextButton.innerText = 'Next Question';
    nextButton.classList.add('next-button');
    nextButton.style.display = 'none'; // Hide initially
    nextButton.onclick = () => {
        currentQuestionIndex++;
        displayQuestion();
        updateProgressBar();  // Update progress bar after moving to the next question
    };
    quizContainer.appendChild(nextButton);
}

function showResults() {
    const quizContainer = document.getElementById('quiz-container');
    const resultsContainer = document.getElementById('results-container');
    const scoreDisplay = document.getElementById('score');
    const messageDisplay = document.getElementById('result-message');

    quizContainer.style.display = 'none';
    resultsContainer.style.display = 'block';

    scoreDisplay.innerText = score;

    // Display a message based on the score
    if (score === 5) {
        messageDisplay.innerText = "Excellent! You have a perfect score!";
        messageDisplay.className = "result-message perfect";
        confetti({
            particleCount: 100,
            spread: 70,
            origin: { y: 0.6 }
        });
    } else if (score >= 3) {
        messageDisplay.innerText = "Good job! You know a lot about deepfakes.";
        messageDisplay.className = "result-message good";
    } else {
        messageDisplay.innerText = "Keep studying! You'll get there!";
        messageDisplay.className = "result-message try-again";
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const draggables = document.querySelectorAll('.draggable');
    const dropzones = document.querySelectorAll('.dropzone');
    const scoreElement = document.getElementById('score');
    const countdownTimer = document.getElementById('countdown-timer');
    const gameSection = document.getElementById('game-section');
    const startButton = document.getElementById('start-button');
    const deck = document.querySelector('.stacked-images');
    let score = 0;
    let timeLeft = 15;  // Set game duration to 15 seconds
    let currentDraggedElement = null;
    let gameRunning = false;
    let usedRotations = [];
    let stackIndex = 1;
    let timerInterval;
    let isCardMoving = false;
    const MAX_CARDS = 10; // Limit the game to 10 cards
    const CARD_WIDTH = '240px'; // Fixed width for all cards
    const CARD_HEIGHT = '288px'; // Fixed height for all cards

    function resetDeck() {
        deck.innerHTML = '';  // Clear the deck
        const shuffledDraggables = shuffleArray(Array.from(draggables));  // Shuffle cards
        const selectedDraggables = shuffledDraggables.slice(0, MAX_CARDS);  // Select only 10 cards
        selectedDraggables.forEach(draggable => {
            const randomRotation = generateUniqueRotation();
            draggable.style.transform = `rotate(${randomRotation}deg)`;
            draggable.style.position = 'absolute';
            draggable.style.left = '';  // Reset left position
            draggable.style.top = '';   // Reset top position
            draggable.style.width = CARD_WIDTH; // Set fixed width
            draggable.style.height = CARD_HEIGHT; // Set fixed height
            draggable.style.zIndex = stackIndex++;  // Ensure stacking order
            draggable.setAttribute('draggable', 'true');  // Ensure draggable attribute is set
            deck.appendChild(draggable);  // Append card back to the deck
        });
    }

    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    function generateUniqueRotation() {
        let rotation;
        do {
            rotation = Math.floor(Math.random() * 30 - 15);
        } while (usedRotations.includes(rotation));

        usedRotations.push(rotation);
        if (usedRotations.length > draggables.length) {
            usedRotations.shift();
        }
        return rotation;
    }

    function startGame() {
        resetDeck();  // Reset the deck at the start of the game
        score = 0;
        scoreElement.textContent = score;
        timeLeft = 15;  // Reset timer to 15 seconds
        countdownTimer.textContent = `Time Left: ${timeLeft}s`;
        countdownTimer.style.display = "block";  // Ensure the timer is visible during the game
        gameSection.style.display = "block";
        startTimer();
    }

    function startTimer() {
        clearInterval(timerInterval);  // Clear any existing timer interval
        timerInterval = setInterval(() => {
            timeLeft -= 1;
            countdownTimer.textContent = `Time Left: ${timeLeft}s`;
            if (timeLeft <= 0) {
                clearInterval(timerInterval);
                endGame();
            }
        }, 1000);
    }

    function endGame() {
        gameRunning = false;
        clearInterval(timerInterval);  // Stop the timer when the game ends
        countdownTimer.style.display = "none";  // Hide the timer after the game ends
        draggables.forEach(draggable => draggable.setAttribute('draggable', 'false'));

        let scoreMessage = '';
        let resultsTitleText = '';
        let imageUrl = '';

        if (score >= 8) {
            resultsTitleText = '"Deepfake Master"';
            scoreMessage = 'Amazing work! You have a keen eye for detail. Nothing gets past you!';
            imageUrl = "static/images/titles/title3.webp";
        } else if (score >= 4) {
            resultsTitleText = '"Cyber detective-in-training"';
            scoreMessage = 'Not bad for a rookie. Try looking harder for details that are either too unnatural or too symmetrical. The drip Pope might get the better of you yet just now.';
            imageUrl = "static/images/titles/title2.webp";
        } else {
            resultsTitleText = '"Deepfake Novice"';
            scoreMessage = 'Keep practicing! You\'ll get better at spotting the fakes. Remember, not everything is as it seems.';
            imageUrl = "static/images/titles/title1.webp";
        }

        setTimeout(() => {
            // Update the results section with the score and messages
            document.getElementById('results-title').textContent = resultsTitleText;
            document.getElementById('results-score').textContent = `Your score: ${score}`;
            document.getElementById('results-message').textContent = scoreMessage;
            document.getElementById('results-image').src = imageUrl; // Set the image source based on the score
            
            // Hide the game section and show the results section
            gameSection.style.display = 'none';
            document.getElementById('results-section').style.display = 'block';
        }, 500);
    }

    // Restart the game when the user clicks the restart button
    document.getElementById('restart-button').addEventListener('click', () => {
        resetGame();
        gameSection.style.display = 'none';  // Hide the game section when resetting
        startButton.disabled = false; // Re-enable the start button for the next round
    });

    function resetGame() {
        gameRunning = false;
        score = 0;
        stackIndex = 1;
        scoreElement.textContent = score;
        clearInterval(timerInterval);

        // Reset all draggables
        draggables.forEach(draggable => {
            draggable.classList.remove('correct', 'discard');
            draggable.style.transition = 'none';
            draggable.style.position = '';
            draggable.style.left = '';
            draggable.style.top = '';
            draggable.style.zIndex = '';
            draggable.style.width = CARD_WIDTH; // Ensure fixed width
            draggable.style.height = CARD_HEIGHT; // Ensure fixed height
            draggable.style.transform = `rotate(${generateUniqueRotation()}deg)`;
            draggable.setAttribute('draggable', 'true');  // Ensure draggable attribute is set

            // Remove event listeners to avoid duplicates
            draggable.removeEventListener('mousedown', dragStart);

            // Reattach event listeners
            draggable.addEventListener('mousedown', dragStart);
        });

        // Reset the deck
        resetDeck();

        // Hide results section
        document.getElementById('results-section').style.display = 'none';

        // Reset dropzones
        dropzones.forEach(dropzone => {
            dropzone.innerHTML = '<h2>' + dropzone.querySelector('h2').textContent + '</h2>';
            dropzone.classList.remove('correct-dropzone', 'incorrect-dropzone');
        });

        // Hide the timer after resetting
        countdownTimer.style.display = 'none';

        // Reset the game state
        isCardMoving = false;
        currentDraggedElement = null;
    }

    startButton.addEventListener('click', () => {
        if (!gameRunning) {
            gameRunning = true;
            startButton.disabled = true;
            countdownTimer.style.display = 'block'; // Show the countdown timer
            let countdown = 3;
            countdownTimer.textContent = countdown;
            const countdownInterval = setInterval(() => {
                countdown -= 1;
                countdownTimer.textContent = countdown; // Display the countdown
                if (countdown === 0) {
                    clearInterval(countdownInterval);
                    startGame();
                }
            }, 1000);
        }
    });

    // Drag and drop functions
    function dragStart(e) {
        if (isCardMoving || !gameRunning) return;  // Prevent new drag event if a card is currently moving or game is not running
        
        isCardMoving = true;  // Set moving flag to true
        currentDraggedElement = e.target; 
        currentDraggedElement.classList.add('active'); // Add the active class on drag start
    
        const rect = currentDraggedElement.getBoundingClientRect();
        shiftX = e.clientX - rect.left;
        shiftY = e.clientY - rect.top;
    
        currentDraggedElement.style.transform = 'rotate(0deg)';
        currentDraggedElement.style.position = 'absolute';
        currentDraggedElement.style.zIndex = '1000'; 
        document.body.append(currentDraggedElement);
    
        moveAt(e.pageX, e.pageY);
    
        function moveAt(pageX, pageY) {
            if (currentDraggedElement) {
                currentDraggedElement.style.left = pageX - shiftX + 'px';
                currentDraggedElement.style.top = pageY - shiftY + 'px';
            }
        }
    
        document.addEventListener('mousemove', moveAt);
    
        currentDraggedElement.onmouseup = function () {
            document.removeEventListener('mousemove', moveAt);
            currentDraggedElement.onmouseup = null;

            if (!currentDraggedElement.classList.contains('correct') && 
                !currentDraggedElement.classList.contains('discard')) {
                resetCard(currentDraggedElement);  // Reset card if not placed in a dropzone
            }

            currentDraggedElement.classList.remove('active'); // Remove the active class when drag ends
            setTimeout(() => isCardMoving = false, 300);  // Allow interaction after a short delay
        };
    }

    function drag(e) {
        if (currentDraggedElement) {
            e.preventDefault();
            currentDraggedElement.style.left = e.pageX - shiftX + 'px';
            currentDraggedElement.style.top = e.pageY - shiftY + 'px';
        }
    }

    function dragEnd(e) {
        if (currentDraggedElement) {
            let elementsAtPoint = document.elementsFromPoint(e.clientX, e.clientY);
            let dropzone = elementsAtPoint.find(el => el.classList.contains('dropzone'));
    
            if (dropzone) {
                const answer = currentDraggedElement.getAttribute('data-answer');
                if (dropzone.id.includes(answer)) {
                    handleCorrectDrop(dropzone);
                } else {
                    handleIncorrectDrop(dropzone);
                }
            } else {
                resetCard(currentDraggedElement); // Reset card if dropped outside of any dropzone
            }
    
            currentDraggedElement.classList.remove('active');
            currentDraggedElement = null; // Ensure the reference to the dragged element is cleared
            isCardMoving = false; // Reset the card moving flag
        }
    }

    function handleCorrectDrop(dropzone) {
        const dropzoneRect = dropzone.getBoundingClientRect();
    
        // Calculate the position to center the card in the dropzone and slightly lower it
        const centerX = (dropzoneRect.width - parseInt(CARD_WIDTH)) / 2;
        const lowerY = (dropzoneRect.height - parseInt(CARD_HEIGHT)) / 2 + 25; // Adjust this value for how much lower you want the card
    
        // Set the card's position to the calculated center and lower position
        currentDraggedElement.style.position = 'absolute';
        currentDraggedElement.style.left = `${centerX}px`;
        currentDraggedElement.style.top = `${lowerY}px`;
        currentDraggedElement.style.width = CARD_WIDTH; // Ensure fixed width
        currentDraggedElement.style.height = CARD_HEIGHT; // Ensure fixed height
        currentDraggedElement.style.zIndex = stackIndex++;
        currentDraggedElement.style.transform = ''; // Remove rotation if any
        dropzone.appendChild(currentDraggedElement);
        currentDraggedElement.classList.add('correct');
        dropzone.classList.add('correct-dropzone');
    
        setTimeout(() => {
            dropzone.classList.remove('correct-dropzone');
        }, 500);
    
        score += 1;
        scoreElement.textContent = score;
        currentDraggedElement.removeEventListener('mousedown', dragStart);
        currentDraggedElement.setAttribute('draggable', 'false');
    
        // Check if all cards are placed or removed
        checkEndGameCondition();
    }
    
    
    function handleIncorrectDrop(dropzone) {
        currentDraggedElement.classList.add('discard'); // Add a discard animation class
    
        // Optional: Play a sound or add a delay before removal to emphasize the discard
        setTimeout(() => {
            // Safely remove the card after a brief delay to ensure the drag operation has fully ended
            if (currentDraggedElement) {
                currentDraggedElement.remove(); // Remove the card from the DOM
                currentDraggedElement = null; // Clear the reference to the dragged element
            }
    
            checkEndGameCondition(); // Check if the game should end after removing the card
        }, 500);  // The delay ensures that the drag end event has fully completed before the card is removed
    
        dropzone.classList.add('incorrect-dropzone');
    
        setTimeout(() => {
            dropzone.classList.remove('incorrect-dropzone');
        }, 500);
    }

    function checkEndGameCondition() {
        // Select all draggable elements that are still in the deck or on the page
        const remainingCards = document.querySelectorAll('.stacked-images .draggable, .dropzone .draggable');
    
        // If no cards are left, end the game
        if (remainingCards.length === 0) {
            endGame();
        }
    }
    

    function resetCard(card) {
        if (card) {
            // Return the card to its original position in the deck
            card.style.transition = 'none';
            card.style.position = '';
            card.style.left = '';
            card.style.top = '';
            card.style.zIndex = '';
            card.style.width = CARD_WIDTH; // Ensure fixed width
            card.style.height = CARD_HEIGHT; // Ensure fixed height
            card.style.transform = `rotate(${generateUniqueRotation()}deg)`;

            setTimeout(() => {
                card.style.transition = 'transform 0.3s ease, box-shadow 0.3s ease';
            }, 10);

            // Append card back to the deck
            deck.appendChild(card);
        }
    }

    // Attach event listeners to draggables
    attachDragListeners();

    function attachDragListeners() {
        draggables.forEach(draggable => {
            draggable.addEventListener('mousedown', dragStart);
        });

        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', dragEnd);
    }
});

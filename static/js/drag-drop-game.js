// Game data and state
        const images = [
            { src: 'static/images/dragdropgame/sample1.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample2.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample3.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample4.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample5.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample6.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample7.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample8.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample9.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample10.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample11.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample12.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample13.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample14.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample15.png', answer: 'real' },
            { src: 'static/images/dragdropgame/sample16.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample17.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample18.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample19.png', answer: 'deepfake' },
            { src: 'static/images/dragdropgame/sample20.png', answer: 'deepfake' }
        ];

        let gameState = {
            currentIndex: 0,
            score: 0,
            timeLeft: 90,
            gameRunning: false,
            shuffledImages: [],
            timer: null,
            isProcessing: false
        };

        // DOM elements
        const heroSection = document.getElementById('hero-section');
        const gameSection = document.getElementById('game-section');
        const resultsSection = document.getElementById('results-section');
        const startBtn = document.getElementById('start-btn');
        const restartBtn = document.getElementById('restart-btn');
        const currentImage = document.getElementById('current-image');
        const scoreDisplay = document.getElementById('score');
        const timerDisplay = document.getElementById('timer');
        const progressDisplay = document.getElementById('progress');
        const imageCounter = document.getElementById('image-counter');
        const btnReal = document.getElementById('btn-real');
        const btnDeepfake = document.getElementById('btn-deepfake');
        const feedback = document.getElementById('feedback');
        const feedbackIcon = document.getElementById('feedback-icon');
        const feedbackMessage = document.getElementById('feedback-message');
        const finalScore = document.getElementById('final-score');
        const accuracy = document.getElementById('accuracy');
        const resultsTitle = document.getElementById('results-title');
        const resultsIcon = document.getElementById('results-icon');
        const achievementMessage = document.getElementById('achievement-message');

        // Utility functions
        function shuffleArray(array) {
            const shuffled = [...array];
            for (let i = shuffled.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
            }
            return shuffled;
        }

        function showSection(section) {
            heroSection.style.display = 'none';
            gameSection.style.display = 'none';
            resultsSection.style.display = 'none';
            
            section.style.display = section === resultsSection ? 'flex' : 'block';
            
            if (section === gameSection) {
                setTimeout(() => {
                    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }, 100);
            }
        }

        function updateDisplay() {
            scoreDisplay.textContent = gameState.score;
            timerDisplay.textContent = `${gameState.timeLeft}s`;
            progressDisplay.textContent = `${gameState.currentIndex + 1} of ${gameState.shuffledImages.length}`;
            imageCounter.textContent = `${gameState.currentIndex + 1} / ${gameState.shuffledImages.length}`;
            
            // Update timer color based on remaining time
            if (gameState.timeLeft <= 10) {
                timerDisplay.style.color = '#ef4444';
            } else if (gameState.timeLeft <= 20) {
                timerDisplay.style.color = '#f59e0b';
            } else {
                timerDisplay.style.color = '';
            }
        }

        function loadCurrentImage() {
            if (gameState.currentIndex < gameState.shuffledImages.length) {
                const currentImageData = gameState.shuffledImages[gameState.currentIndex];
                // For demo purposes, using placeholder images
                currentImage.src = currentImageData.src;
                currentImage.alt = `Image ${gameState.currentIndex + 1}`;
                currentImage.classList.add('slide-in');
                
                setTimeout(() => {
                    currentImage.classList.remove('slide-in');
                }, 500);
            }
        }

        function showFeedback(isCorrect, message) {
            feedback.className = `feedback ${isCorrect ? 'correct' : 'incorrect'} show`;
            feedbackIcon.textContent = isCorrect ? '✅' : '❌';
            feedbackMessage.textContent = message;
            
            setTimeout(() => {
                feedback.classList.remove('show');
            }, 1500);
        }

        function startTimer() {
            gameState.timer = setInterval(() => {
                gameState.timeLeft--;
                updateDisplay();
                
                if (gameState.timeLeft <= 0) {
                    endGame();
                }
            }, 1000);
        }

        function handleChoice(choice) {
            if (!gameState.gameRunning || gameState.isProcessing) return;
            
            gameState.isProcessing = true;
            btnReal.disabled = true;
            btnDeepfake.disabled = true;
            
            const currentImageData = gameState.shuffledImages[gameState.currentIndex];
            const isCorrect = choice === currentImageData.answer;
            
            if (isCorrect) {
                gameState.score++;
                showFeedback(true, 'Correct!');
                
                // Confetti for correct answer
                if (typeof confetti !== 'undefined') {
                    confetti({
                        particleCount: 50,
                        spread: 70,
                        origin: { y: 0.6 }
                    });
                }
            } else {
                showFeedback(false, `Wrong! It was ${currentImageData.answer}`);
            }
            
            gameState.currentIndex++;
            updateDisplay();

            // Check if game should end immediately after updating the index
            if (gameState.currentIndex >= gameState.shuffledImages.length) {
                setTimeout(() => {
                    endGame();
                }, 2000);
                return;
            }

            setTimeout(() => {
                if (gameState.timeLeft <= 0) {
                    endGame();
                } else {
                    loadCurrentImage();
                    gameState.isProcessing = false;
                    btnReal.disabled = false;
                    btnDeepfake.disabled = false;
                }
            }, 2000);
        }

        function startGame() {
            gameState = {
                currentIndex: 0,
                score: 0,
                timeLeft: 90,
                gameRunning: true,
                shuffledImages: shuffleArray(images),
                timer: null,
                isProcessing: false
            };
            
            showSection(gameSection);
            updateDisplay();
            loadCurrentImage();
            
            // Countdown before starting
            let countdown = 3;
            btnReal.disabled = true;
            btnDeepfake.disabled = true;
            
            const countdownInterval = setInterval(() => {
                timerDisplay.textContent = countdown;
                timerDisplay.style.fontSize = '2rem';
                timerDisplay.style.color = '#667eea';
                
                countdown--;
                
                if (countdown < 0) {
                    clearInterval(countdownInterval);
                    timerDisplay.style.fontSize = '';
                    timerDisplay.style.color = '';
                    btnReal.disabled = false;
                    btnDeepfake.disabled = false;
                    startTimer();
                }
            }, 1000);
        }

        function endGame() {
            gameState.gameRunning = false;
            if (gameState.timer) {
                clearInterval(gameState.timer);
            }
            
            btnReal.disabled = true;
            btnDeepfake.disabled = true;
            
            setTimeout(() => {
                showResults();
            }, 1000);
        }

        function showResults() {
            const totalImages = gameState.shuffledImages.length;
            const accuracyPercent = Math.round((gameState.score / totalImages) * 100);
            
            finalScore.textContent = `${gameState.score}/${totalImages}`;
            accuracy.textContent = `${accuracyPercent}%`;
            
            // Determine achievement level
            if (accuracyPercent >= 90) {
                resultsTitle.textContent = 'Deepfake Master! 🏆';
                resultsIcon.textContent = '🏆';
                achievementMessage.textContent = 'Incredible! You have an eagle eye for spotting deepfakes. Your skills are exceptional!';
                
                // Celebration confetti
                if (typeof confetti !== 'undefined') {
                    confetti({
                        particleCount: 150,
                        spread: 70,
                        origin: { y: 0.6 }
                    });
                }
            } else if (accuracyPercent >= 70) {
                resultsTitle.textContent = 'Cyber Detective! 🕵️';
                resultsIcon.textContent = '🕵️';
                achievementMessage.textContent = 'Well done! You\'re getting good at this. With a bit more practice, you\'ll be a master detective.';
            } else if (accuracyPercent >= 50) {
                resultsTitle.textContent = 'Deepfake Apprentice 📚';
                resultsIcon.textContent = '📚';
                achievementMessage.textContent = 'Not bad for a start! Keep practicing and studying the techniques to improve your detection skills.';
            } else {
                resultsTitle.textContent = 'Keep Learning! 💪';
                resultsIcon.textContent = '💪';
                achievementMessage.textContent = 'Everyone starts somewhere! Study the educational resources and try again. You\'ll improve with practice!';
            }
            
            showSection(resultsSection);
        }

        function resetGame() {
            window.location.reload();
        }

        // Event listeners
        startBtn.addEventListener('click', startGame);
        restartBtn.addEventListener('click', resetGame);

        btnReal.addEventListener('click', () => handleChoice('real'));
        btnDeepfake.addEventListener('click', () => handleChoice('deepfake'));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (!gameState.gameRunning || gameState.isProcessing) return;
            
            if (e.key === '1' || e.key.toLowerCase() === 'r') {
                handleChoice('real');
            } else if (e.key === '2' || e.key.toLowerCase() === 'd') {
                handleChoice('deepfake');
            }
        });

        // Prevent context menu on images
        currentImage.addEventListener('contextmenu', (e) => {
            e.preventDefault();
        });

        // Initialize
        updateDisplay();
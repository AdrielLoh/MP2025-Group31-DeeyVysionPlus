document.addEventListener('DOMContentLoaded', () => {
    const gameCards = document.querySelectorAll('.game-card');
    const dropzones = document.querySelectorAll('.drop-zone');
    const scoreElement = document.getElementById('score');
    const countdownTimer = document.getElementById('countdown-timer');
    const remainingCount = document.getElementById('remaining-count');
    const gameSection = document.getElementById('game-section');
    const resultsSection = document.getElementById('results-section');
    const startButton = document.getElementById('start-button');
    const restartButton = document.getElementById('restart-button');
    const cardStack = document.getElementById('card-stack');
    
    let score = 0;
    let timeLeft = 30;
    let currentDraggedElement = null;
    let gameRunning = false;
    let usedRotations = [];
    let stackIndex = 1;
    let timerInterval;
    let isCardMoving = false;
    let totalCards = 20;
    let processedCards = 0;
    let shiftX, shiftY;
    let originalPosition = {};

    function initializeGame() {
        resetGameState();
        shuffleCards();
        setupEventListeners();
    }

    function resetGameState() {
        score = 0;
        timeLeft = 30;
        processedCards = 0;
        stackIndex = 1;
        gameRunning = false;
        isCardMoving = false;
        currentDraggedElement = null;
        usedRotations = [];
        originalPosition = {};
        
        scoreElement.textContent = score;
        countdownTimer.textContent = `${timeLeft}s`;
        remainingCount.textContent = totalCards;
        
        clearInterval(timerInterval);
    }

    function shuffleCards() {
        const cards = Array.from(gameCards);
        const shuffledCards = shuffleArray(cards);
        
        cardStack.innerHTML = '';
        
        shuffledCards.forEach((card, index) => {
            const rotation = generateUniqueRotation();
            card.style.transform = `rotate(${rotation}deg)`;
            card.style.position = 'absolute';
            card.style.left = '0';
            card.style.top = '0';
            card.style.zIndex = totalCards - index;
            card.classList.remove('dragging', 'correct', 'incorrect');
            card.setAttribute('draggable', 'true');
            cardStack.appendChild(card);
        });
    }

    function shuffleArray(array) {
        const shuffled = [...array];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    function generateUniqueRotation() {
        let rotation;
        do {
            rotation = Math.floor(Math.random() * 20 - 10); // -10 to 10 degrees
        } while (usedRotations.includes(rotation) && usedRotations.length < 20);
        
        usedRotations.push(rotation);
        if (usedRotations.length > 20) {
            usedRotations.shift();
        }
        return rotation;
    }

    function setupEventListeners() {
        gameCards.forEach(card => {
            card.addEventListener('dragstart', handleDragStart);
            card.addEventListener('mousedown', handleMouseDown);
        });

        dropzones.forEach(zone => {
            zone.addEventListener('dragover', handleDragOver);
            zone.addEventListener('drop', handleDrop);
            zone.addEventListener('dragenter', handleDragEnter);
            zone.addEventListener('dragleave', handleDragLeave);
        });

        document.addEventListener('mousemove', handleMouseMove);
        document.addEventListener('mouseup', handleMouseUp);
    }

    function startGame() {
        gameRunning = true;
        gameSection.style.display = 'block';
        resultsSection.style.display = 'none';
        startButton.disabled = true;

        // Add this scroll functionality
        setTimeout(() => {
            gameSection.scrollIntoView({ 
                behavior: 'smooth',
                block: 'start'
            });
        }, 100); // Small delay to ensure the section is displayed first
        
        // Countdown before game starts
        let countdown = 5;
        countdownTimer.textContent = countdown;
        countdownTimer.style.fontSize = '2rem';
        countdownTimer.style.color = '#667eea';
        
        const countdownInterval = setInterval(() => {
            countdown--;
            if (countdown > 0) {
                countdownTimer.textContent = countdown;
                // Add pulse animation
                countdownTimer.style.transform = 'scale(1.2)';
                setTimeout(() => {
                    countdownTimer.style.transform = 'scale(1)';
                }, 200);
            } else {
                clearInterval(countdownInterval);
                countdownTimer.textContent = `${timeLeft}s`;
                countdownTimer.style.fontSize = '';
                countdownTimer.style.color = '';
                startGameTimer();
            }
        }, 1000);
    }

    function startGameTimer() {
        timerInterval = setInterval(() => {
            timeLeft--;
            countdownTimer.textContent = `${timeLeft}s`;
            
            // Add warning colors as time runs out
            if (timeLeft <= 10) {
                countdownTimer.style.color = '#ef4444';
                countdownTimer.style.fontWeight = '700';
            } else if (timeLeft <= 20) {
                countdownTimer.style.color = '#f59e0b';
            }
            
            if (timeLeft <= 0) {
                endGame();
            }
        }, 1000);
    }

    function handleDragStart(e) {
        if (!gameRunning || isCardMoving) {
            e.preventDefault();
            return;
        }
        
        currentDraggedElement = e.target;
        currentDraggedElement.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
    }

    function handleMouseDown(e) {
        if (!gameRunning || isCardMoving) return;
        
        e.preventDefault(); // Prevent default drag behavior
        isCardMoving = true;
        currentDraggedElement = e.target;
        currentDraggedElement.classList.add('dragging');
        
        // Get current position before any changes
        const rect = currentDraggedElement.getBoundingClientRect();
        
        // Store original position and styling
        originalPosition = {
            left: currentDraggedElement.style.left,
            top: currentDraggedElement.style.top,
            position: currentDraggedElement.style.position,
            zIndex: currentDraggedElement.style.zIndex,
            transform: currentDraggedElement.style.transform
        };
        
        // Calculate offset from mouse to element's top-left corner
        shiftX = e.clientX - rect.left;
        shiftY = e.clientY - rect.top;
        
        // Switch to fixed positioning while maintaining current visual position
        currentDraggedElement.style.position = 'fixed';
        currentDraggedElement.style.left = rect.left + 'px';
        currentDraggedElement.style.top = rect.top + 'px';
        currentDraggedElement.style.zIndex = '1000';
        currentDraggedElement.style.pointerEvents = 'none';
        currentDraggedElement.style.transform = 'rotate(0deg) scale(1.05)';
        
        // Prevent text selection
        document.body.style.userSelect = 'none';
    }

    function handleMouseMove(e) {
        if (currentDraggedElement && isCardMoving) {
            moveAt(e.clientX, e.clientY);
            
            // Add visual feedback for drop zones
            const elementsAtPoint = document.elementsFromPoint(e.clientX, e.clientY);
            const dropzone = elementsAtPoint.find(el => el.classList.contains('drop-zone'));
            
            // Remove drag-over class from all zones
            dropzones.forEach(zone => zone.classList.remove('drag-over'));
            
            // Add drag-over class to current zone
            if (dropzone) {
                dropzone.classList.add('drag-over');
            }
        }
    }

    function handleMouseUp(e) {
        if (!currentDraggedElement || !isCardMoving) return;
        
        // Re-enable pointer events and text selection
        currentDraggedElement.style.pointerEvents = '';
        document.body.style.userSelect = '';
        
        const elementsAtPoint = document.elementsFromPoint(e.clientX, e.clientY);
        const dropzone = elementsAtPoint.find(el => el.classList.contains('drop-zone'));
        
        if (dropzone) {
            handleCardDrop(dropzone);
        } else {
            resetCard();
        }
        
        cleanup();
    }

    function moveAt(clientX, clientY) {
        if (currentDraggedElement) {
            currentDraggedElement.style.left = (clientX - shiftX) + 'px';
            currentDraggedElement.style.top = (clientY - shiftY) + 'px';
        }
    }

    function handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
    }

    function handleDragEnter(e) {
        if (currentDraggedElement) {
            e.target.closest('.drop-zone').classList.add('drag-over');
        }
    }

    function handleDragLeave(e) {
        const zone = e.target.closest('.drop-zone');
        if (zone && !zone.contains(e.relatedTarget)) {
            zone.classList.remove('drag-over');
        }
    }

    function handleDrop(e) {
        e.preventDefault();
        const dropzone = e.target.closest('.drop-zone');
        handleCardDrop(dropzone);
        cleanup();
    }

    function handleCardDrop(dropzone) {
        if (!dropzone || !currentDraggedElement) return;
        
        dropzone.classList.remove('drag-over');
        
        const cardAnswer = currentDraggedElement.getAttribute('data-answer');
        const zoneAnswer = dropzone.getAttribute('data-answer');
        
        if (cardAnswer === zoneAnswer) {
            handleCorrectDrop(dropzone);
        } else {
            handleIncorrectDrop(dropzone);
        }
    }

    function handleCorrectDrop(dropzone) {
        score++;
        processedCards++;
        
        // Position the card in the center of the drop zone
        const dropzoneRect = dropzone.getBoundingClientRect();
        const centerX = dropzoneRect.left + dropzoneRect.width / 2;
        const centerY = dropzoneRect.top + dropzoneRect.height / 2;
        
        currentDraggedElement.style.left = (centerX - currentDraggedElement.offsetWidth / 2) + 'px';
        currentDraggedElement.style.top = (centerY - currentDraggedElement.offsetHeight / 2) + 'px';
        
        // Visual feedback
        currentDraggedElement.classList.add('correct');
        dropzone.classList.add('correct-drop');
        currentDraggedElement.style.transform = 'scale(0.8)';
        currentDraggedElement.style.opacity = '0.8';
        
        // Confetti effect for correct answers
        if (typeof confetti !== 'undefined') {
            confetti({
                particleCount: 30,
                spread: 45,
                origin: { 
                    x: centerX / window.innerWidth,
                    y: centerY / window.innerHeight
                }
            });
        }
        
        // Fade out and remove the card
        setTimeout(() => {
            if (currentDraggedElement) {
                currentDraggedElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                currentDraggedElement.style.opacity = '0';
                currentDraggedElement.style.transform = 'scale(0.5)';
                
                setTimeout(() => {
                    dropzone.classList.remove('correct-drop');
                    if (currentDraggedElement && currentDraggedElement.parentNode) {
                        currentDraggedElement.remove();
                    }
                }, 300);
            } else {
                dropzone.classList.remove('correct-drop');
            }
        }, 200);
        
        updateUI();
        checkGameEnd();
    }

    function handleIncorrectDrop(dropzone) {
        processedCards++;
        
        // Position the card in the center of the drop zone
        const dropzoneRect = dropzone.getBoundingClientRect();
        const centerX = dropzoneRect.left + dropzoneRect.width / 2;
        const centerY = dropzoneRect.top + dropzoneRect.height / 2;
        
        currentDraggedElement.style.left = (centerX - currentDraggedElement.offsetWidth / 2) + 'px';
        currentDraggedElement.style.top = (centerY - currentDraggedElement.offsetHeight / 2) + 'px';
        
        // Visual feedback
        currentDraggedElement.classList.add('incorrect');
        dropzone.classList.add('incorrect-drop');
        currentDraggedElement.style.transform = 'scale(0.9)';
        currentDraggedElement.style.opacity = '0.8';
        
        // Shake animation for incorrect answers
        setTimeout(() => {
            if (currentDraggedElement) {
                currentDraggedElement.style.animation = 'shake 0.5s ease-in-out';
            }
        }, 100);
        
        // Fade out and remove the card
        setTimeout(() => {
            if (currentDraggedElement) {
                currentDraggedElement.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                currentDraggedElement.style.opacity = '0';
                currentDraggedElement.style.transform = 'scale(0.5)';
                
                setTimeout(() => {
                    dropzone.classList.remove('incorrect-drop');
                    if (currentDraggedElement && currentDraggedElement.parentNode) {
                        currentDraggedElement.remove();
                    }
                }, 300);
            } else {
                dropzone.classList.remove('incorrect-drop');
            }
        }, 600);
        
        updateUI();
        checkGameEnd();
    }

    function resetCard() {
        if (!currentDraggedElement) return;
        
        // Restore original position and styling
        currentDraggedElement.style.position = originalPosition.position;
        currentDraggedElement.style.left = originalPosition.left;
        currentDraggedElement.style.top = originalPosition.top;
        currentDraggedElement.style.zIndex = originalPosition.zIndex;
        currentDraggedElement.style.transform = originalPosition.transform;
        
        // Ensure the card is back in the stack
        if (!cardStack.contains(currentDraggedElement)) {
            cardStack.appendChild(currentDraggedElement);
        }
    }

    function cleanup() {
        if (currentDraggedElement) {
            currentDraggedElement.classList.remove('dragging');
            // Remove drag-over from all zones
            dropzones.forEach(zone => zone.classList.remove('drag-over'));
            currentDraggedElement = null;
        }
        isCardMoving = false;
        originalPosition = {};
        
        // Re-enable text selection
        document.body.style.userSelect = '';
    }

    function updateUI() {
        scoreElement.textContent = score;
        remainingCount.textContent = totalCards - processedCards;
        
        // Add score animation
        scoreElement.style.transform = 'scale(1.2)';
        scoreElement.style.color = '#10b981';
        setTimeout(() => {
            scoreElement.style.transform = 'scale(1)';
            scoreElement.style.color = '';
        }, 200);
    }

    function checkGameEnd() {
        if (processedCards >= totalCards) {
            endGame();
        }
    }

    function endGame() {
        gameRunning = false;
        clearInterval(timerInterval);
        
        // Disable all cards
        gameCards.forEach(card => {
            card.setAttribute('draggable', 'false');
            card.style.pointerEvents = 'none';
        });
        
        // Clean up any remaining drag state
        cleanup();
        
        setTimeout(() => {
            showResults();
        }, 1000);
    }

    function showResults() {
        gameSection.style.display = 'none';
        resultsSection.style.display = '';
        
        const accuracy = Math.round((score / totalCards) * 100);
        const resultsIcon = document.getElementById('results-icon');
        const resultsTitle = document.getElementById('results-title');
        const resultsScore = document.getElementById('results-score');
        const accuracyPercentage = document.getElementById('accuracy-percentage');
        const resultsBadge = document.getElementById('results-badge');
        const resultsMessage = document.getElementById('results-message');
        
        resultsScore.textContent = score;
        accuracyPercentage.textContent = `${accuracy}%`;
        
        // Determine achievement level
        if (accuracy >= 90) {
            resultsTitle.textContent = 'Deepfake Master! 🏆';
            resultsIcon.textContent = '🏆';
            resultsBadge.src = "static/images/titles/title3.webp";
            resultsMessage.textContent = 'Incredible! You have an eagle eye for spotting deepfakes. Your skills are exceptional!';
            
            // Celebration confetti
            if (typeof confetti !== 'undefined') {
                confetti({
                    particleCount: 150,
                    spread: 70,
                    origin: { y: 0.6 }
                });
            }
        } else if (accuracy >= 70) {
            resultsTitle.textContent = 'Cyber Detective!';
            resultsIcon.textContent = '🕵️';
            resultsBadge.src = "static/images/titles/title2.webp";
            resultsMessage.textContent = 'Well done! You\'re getting good at this. With a bit more practice, you\'ll be a master detective.';
        } else if (accuracy >= 50) {
            resultsTitle.textContent = 'Deepfake Apprentice';
            resultsIcon.textContent = '📚';
            resultsBadge.src = "static/images/titles/title1.webp";
            resultsMessage.textContent = 'Not bad for a start! Keep practicing and studying the techniques to improve your detection skills.';
        } else {
            resultsTitle.textContent = 'Keep Learning!';
            resultsIcon.textContent = '💪';
            resultsBadge.src = "static/images/titles/title1.webp";
            resultsMessage.textContent = 'Everyone starts somewhere! Study the educational resources and try again. You\'ll improve with practice!';
        }
    }

    // Event Listeners
    startButton.addEventListener('click', () => {
        if (!gameRunning) {
            startGame();
        }
    });

    restartButton.addEventListener('click', () => {
        resetGameState();
        shuffleCards();
        resultsSection.style.display = 'none';
        startButton.disabled = false;
        
        // Smooth scroll back to top
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    // Prevent context menu on cards during game
    document.addEventListener('contextmenu', (e) => {
        if (e.target.classList.contains('game-card') && gameRunning) {
            e.preventDefault();
        }
    });

    // Initialize the game
    initializeGame();
});
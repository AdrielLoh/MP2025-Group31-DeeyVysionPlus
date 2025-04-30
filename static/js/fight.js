document.addEventListener('DOMContentLoaded', function() {
    const playerHealthBar = document.getElementById('player-health-bar');
    const enemyHealthBar = document.getElementById('enemy-health-bar');
    const playerHpCounter = document.getElementById('player-hp');
    const enemyHpCounter = document.getElementById('enemy-hp');
    const answerButtonsContainer = document.querySelector('.answer-buttons');
    const questionText = document.getElementById('question-text');
    const gameOverScreen = document.getElementById('game-over-screen');
    const gameOverTitle = document.getElementById('game-over-title');
    const gameOverSummary = document.getElementById('game-over-summary');
    const gameOverMessage = document.getElementById('game-over-message');
    const gameOverButtons = document.querySelector('.game-over-buttons');
    const enemyAvatar = document.getElementById('enemy-avatar');
    const timerElement = document.getElementById('timer'); // Timer element

    const maxHealth = 100; // Both player and enemy start with 100 health points
    let playerHealth = maxHealth;
    let enemyHealth = maxHealth;
    let playerPosition = 10; // Track player's position for running animation
    let enemyPosition = 10; // Track enemy's position

    let currentQuestionIndex = 0; // Declare currentQuestionIndex here to ensure it's accessible globally
    let stageQuestions = []; // Declare stageQuestions to be accessible throughout

    const bossNames = {
        1: 'AI Overlord',
        2: 'Fabricator Phantom',
        3: 'Misinformation Master',
        4: 'Identity Shifter',
        5: 'Reality Warper'
    };

    const enemySpriteSheets = {
        1: 'static/images/game/sprites/ai-overlord.png',
        2: 'static/images/game/sprites/fabricator-phantom.png',
        3: 'static/images/game/sprites/misinformation-master.png',
        4: 'static/images/game/sprites/identity-shifter.png',
        5: 'static/images/game/sprites/reality-warper.png'
    };

    const enemyAvatars = {
        1: 'static/images/game/avatars/ai-overlord.webp',
        2: 'static/images/game/avatars/fabricator-phantom.webp',
        3: 'static/images/game/avatars/misinformation-master.webp',
        4: 'static/images/game/avatars/identity-shifter.webp',
        5: 'static/images/game/avatars/reality-warper.webp'
    };

    const stage = parseInt(new URLSearchParams(window.location.search).get('stage'));
    const bossName = bossNames[stage] || 'Enemy';
    const enemySpriteSheetPath = enemySpriteSheets[stage];
    const enemyAvatarPath = enemyAvatars[stage];

    document.getElementById('enemy-name').textContent = bossName;
    enemyAvatar.src = enemyAvatarPath;

    const playerCharacter = document.getElementById('player-character');
    const enemyCharacter = document.getElementById('enemy-character');

    // Canvas and context initialization for player
    const playerCanvas = document.createElement('canvas');
    const playerCtx = playerCanvas.getContext('2d');
    playerCanvas.width = 500;  // Set the canvas to 500px width
    playerCanvas.height = 500; // Set the canvas to 500px height
    playerCharacter.appendChild(playerCanvas);

    // Canvas and context initialization for enemy
    const enemyCanvas = document.createElement('canvas');
    const enemyCtx = enemyCanvas.getContext('2d');
    enemyCanvas.width = 500;
    enemyCanvas.height = 500;
    enemyCharacter.appendChild(enemyCanvas);

    // Disable image smoothing to prevent blurriness
    playerCtx.imageSmoothingEnabled = false;
    enemyCtx.imageSmoothingEnabled = false;

    let playerSpriteSheet;
    let playerAnimationData;
    let enemySpriteSheet;
    let enemyAnimationData;

    // Function to load sprite sheet and JSON data together
    function loadSpriteAndAnimationData(spriteSheetPath, jsonPath, onLoaded) {
        const spriteSheet = new Image();
        let animationData = null;

        // Fetch the JSON data
        fetch(jsonPath)
            .then(response => response.json())
            .then(data => {
                animationData = processAnimationData(data.frames);

                // Ensure idle animation data exists
                if (!animationData['idle']) {
                    throw new Error("Idle animation not found in JSON data.");
                }

                // Load the sprite sheet
                spriteSheet.onload = function() {
                    onLoaded(spriteSheet, animationData);
                };
                spriteSheet.src = spriteSheetPath;
            })
            .catch(error => console.error('Error loading JSON or sprite sheet:', error));
    }

    // Load player data
    loadSpriteAndAnimationData('static/images/game/sprites/agent-dv.png', 'static/json/agent-dv.json', (spriteSheet, animationData) => {
        playerSpriteSheet = spriteSheet;
        playerAnimationData = animationData;
        playIdleAnimation(); // Trigger idle animation after everything is loaded
    });

    // Load enemy data
    loadSpriteAndAnimationData(enemySpriteSheetPath, `static/json/${bossName.toLowerCase().replace(/ /g, '-')}.json`, (spriteSheet, animationData) => {
        enemySpriteSheet = spriteSheet;
        enemyAnimationData = animationData;
        playEnemyIdleAnimation(); // Trigger enemy idle animation after everything is loaded
    });

    // Disable buttons
    function disableAnswerButtons() {
        const buttons = document.querySelectorAll('.answer-button');
        buttons.forEach(button => {
            button.disabled = true;
        });
    }
    
    // Enable buttons
    function enableAnswerButtons() {
        const buttons = document.querySelectorAll('.answer-button');
        buttons.forEach(button => {
            button.disabled = false;
        });
    }

    function processAnimationData(frames) {
        const animationData = {};

        for (const [fileName, frameData] of Object.entries(frames)) {
            const animationType = fileName.split('_')[2]; // e.g., "attack", "idle", "run"

            if (!animationData[animationType]) {
                animationData[animationType] = [];
            }

            animationData[animationType].push({
                frame: frameData.frame,
                offset: frameData.offset,
                rotated: frameData.rotated,
                sourceColorRect: frameData.sourceColorRect,
                sourceSize: frameData.sourceSize
            });
        }

        return animationData;
    }

    let currentPlayerAnimation; // Track current player's animation
    let currentEnemyAnimation;  // Track current enemy's animation
    
    function stopPlayerAnimation() {
        if (currentPlayerAnimation) {
            cancelAnimationFrame(currentPlayerAnimation); // Cancel the current animation frame
            currentPlayerAnimation = null;
        }
        isPlayerIdle = false; // Ensure the idle flag is reset
    }
    
    function stopEnemyAnimation() {
        if (currentEnemyAnimation) {
            cancelAnimationFrame(currentEnemyAnimation); // Cancel the current animation frame
            currentEnemyAnimation = null;
        }
        isEnemyIdle = false; // Ensure the idle flag is reset
    }
    
    let isPlayerIdle = false;
    let isEnemyIdle = false;
    
    function playIdleAnimation() {
        if (!isPlayerIdle) {
            stopPlayerAnimation();  // Ensure the previous idle animation is stopped
            isPlayerIdle = true;     // Set idle flag
            playerPosition = 10;     // Reset player position
            const idleFrameRate = 15; // Set to 5 FPS for slower idle animation
            playAnimation(playerSpriteSheet, playerAnimationData, playerCtx, playerCanvas, 'idle', idleFrameRate, true, null, false, true);
        }
    }
    
    function playEnemyIdleAnimation() {
        if (!isEnemyIdle) {
            stopEnemyAnimation();  // Ensure the previous idle animation is stopped
            isEnemyIdle = true;    // Set idle flag
            enemyPosition = 10;    // Reset enemy position
            const idleFrameRate = 15; // Set to 5 FPS for slower idle animation
            playAnimation(enemySpriteSheet, enemyAnimationData, enemyCtx, enemyCanvas, 'idle', idleFrameRate, true, null, true, false);
        }
    }
    
    function playRunAnimation(callback) {
        stopPlayerAnimation();  // Stop any existing animation, including idle
        isPlayerIdle = false;   // Ensure idle state is false
        let runPosition = 0;
        const runDistance = 100; // 90% of the container's width
        const frameRate = 45;   // Set to 45 FPS for smoother running animation
        const runStep = runDistance / (playerAnimationData['run'].length); // Adjust step size
    
        disableAnswerButtons(); // Disable buttons during animation
    
        playAnimation(playerSpriteSheet, playerAnimationData, playerCtx, playerCanvas, 'run', frameRate, false, function() {
            moveCharacter(); // Move character after run animation
        });
    
        function moveCharacter() {
            const runMovementInterval = setInterval(function() {
                runPosition += runStep;
                playerPosition += runStep;
                playerCharacter.style.left = `${playerPosition}%`;
    
                if (runPosition >= runDistance) {
                    clearInterval(runMovementInterval);
                    stopPlayerAnimation(); // Stop any animation right after running
                    if (callback) callback(); // Trigger the callback (attack animation) only after stopping all animations
                }
            }, 1000 / frameRate); // Adjust based on the new frame rate
        }
    }
    
    function playAttackAnimation(callback) {
        if (!isPlayerIdle) { // Ensure attack animation only triggers if not idle
            stopPlayerAnimation();  // Stop any existing animation, including run and idle
            isPlayerIdle = false;   // Ensure idle state is false
            disableAnswerButtons(); // Disable buttons during animation
        
            playAnimation(playerSpriteSheet, playerAnimationData, playerCtx, playerCanvas, 'attack', 28, false, function() {
                setTimeout(() => {
                    playerCharacter.style.left = '10%'; // Move character back to original position
                    playIdleAnimation(); // Trigger idle animation after attack
                    enableAnswerButtons(); // Re-enable buttons
                    if (callback) callback(); // Ensure callback is called once
                }, 1500); // Adjust timing based on attack animation duration
            });
        }
    }
    
    function playEnemyRunAnimation(callback) {
        stopEnemyAnimation();  // Stop any existing animation, including idle
        isEnemyIdle = false;   // Ensure idle state is false
        let runPosition = 0;
        const runDistance = 100; // 90% of the container's width
        const frameRate = 45;   // Set to 45 FPS for smoother running animation
        const runStep = runDistance / enemyAnimationData['run'].length; // Adjust step size
    
        disableAnswerButtons(); // Disable buttons during animation
    
        playAnimation(enemySpriteSheet, enemyAnimationData, enemyCtx, enemyCanvas, 'run', frameRate, false, function() {
            moveCharacter(); // Move character after run animation frames complete
        }, true, false); // Mirrored = true, isPlayer = false
    
        function moveCharacter() {
            const runMovementInterval = setInterval(function() {
                runPosition += runStep;
                enemyPosition += runStep;
                enemyCharacter.style.right = `${enemyPosition}%`;
    
                if (runPosition >= runDistance) {
                    clearInterval(runMovementInterval);
                    stopEnemyAnimation(); // Stop any ongoing animation after movement
                    if (callback) callback(); // Proceed to attack animation
                }
            }, 1000 / frameRate); // Movement updates at the same frame rate
        }
    }
    
    function playEnemyAttackAnimation(callback) {
        if (!isEnemyIdle) { // Ensure attack animation only triggers if not idle
            stopEnemyAnimation();  // Stop any existing animation, including run and idle
            isEnemyIdle = false;   // Ensure idle state is false
            disableAnswerButtons(); // Ensure buttons remain disabled during attack
        
            playAnimation(enemySpriteSheet, enemyAnimationData, enemyCtx, enemyCanvas, 'attack', 28, false, function() {
                setTimeout(() => {
                    enemyCharacter.style.right = '10%'; // Reset enemy position
                    playEnemyIdleAnimation(); // Return to idle animation after attack
                    enableAnswerButtons(); // Re-enable buttons after sequence completes
                    if (callback) callback(); // Proceed with any additional logic
                }, 1500); // Adjust timing based on attack animation duration
            }, true, false); // Mirrored = true, isPlayer = false
        }
    }
     
    function playAnimation(spriteSheet, animationData, context, canvas, type, frameRate = 24, loop = false, callback = null, mirrored = false, isPlayer = true) {
        console.log(`playAnimation called for type: ${type}`);
        
        if (isPlayer) {
            stopPlayerAnimation(); // Stop any ongoing player animation
        } else {
            stopEnemyAnimation(); // Stop any ongoing enemy animation
        }
    
        // Ensure the previous animation is fully stopped before starting a new one
        cancelAnimationFrame(currentPlayerAnimation);
        cancelAnimationFrame(currentEnemyAnimation);
    
        // Proceed with the new animation
        if (spriteSheet.complete && spriteSheet.naturalWidth !== 0) {
            const frames = animationData[type];
            if (!frames) {
                console.error(`No frames found for animation type: ${type}`);
                return;
            }
    
            let index = 0;
    
            // Calculate the delay between frames based on the frameRate
            let delay = 1000 / frameRate; // Delay in milliseconds per frame
    
            function animate() {
                if (index >= frames.length) {
                    index = 0;
                    if (!loop) {
                        if (callback) {
                            callback();
                        }
                        return;
                    }
                }
    
                const frameData = frames[index];
                const framePos = frameData.frame.match(/\{([\d\s,]+)\}/g)[0].replace(/[{}]/g, '').split(',').map(Number);
                const frameSize = frameData.frame.match(/\{([\d\s,]+)\}/g)[1].replace(/[{}]/g, '').split(',').map(Number);
    
                context.clearRect(0, 0, canvas.width, canvas.height);
    
                if (mirrored) {
                    context.save();
                    context.scale(-1, 1);
                    context.drawImage(spriteSheet, framePos[0], framePos[1], frameSize[0], frameSize[1], -canvas.width, 0, canvas.width, canvas.height);
                    context.restore();
                } else {
                    context.drawImage(spriteSheet, framePos[0], framePos[1], frameSize[0], frameSize[1], 0, 0, canvas.width, canvas.height);
                }
    
                index++;
    
                if (isPlayer) {
                    currentPlayerAnimation = setTimeout(animate, delay);
                } else {
                    currentEnemyAnimation = setTimeout(animate, delay);
                }
            }
    
            animate();
        } else {
            console.error('Sprite sheet not fully loaded. Unable to play animation.');
        }
    }

    // Timer IDs to track the timer for each question and countdown
    let answerTimer;
    let countdown;

    // Time in milliseconds (5 seconds) before auto-selecting wrong answer
    const answerTimeout = 5000;

    function startAnswerTimer() {
        // Clear any existing timers
        clearTimeout(answerTimer);
        clearInterval(countdown);

        let timeRemaining = 5;
        timerElement.textContent = timeRemaining; // Initialize the timer display

        countdown = setInterval(() => {
            timeRemaining--;
            timerElement.textContent = timeRemaining;

            if (timeRemaining <= 0) {
                clearInterval(countdown);
                // Automatically handle as wrong answer
                handleAnswer(null, false, bossName);
            }
        }, 1000);

        // Start a new timer that triggers after the timeout duration
        answerTimer = setTimeout(() => {
            clearInterval(countdown);
            handleAnswer(null, false, bossName);
        }, answerTimeout);
    }

    // Handle answer selection
    function handleAnswer(button, isCorrect, bossName) {
        // Clear the timer since the user has now answered
        clearTimeout(answerTimer);
        clearInterval(countdown);
        
        disableAnswerButtons();

        if (isCorrect) {
            if (button) button.classList.add('correct');
            console.log("Correct answer selected, initiating run and attack animations.");

            playRunAnimation(function() {
                playAttackAnimation(function() {
                    if (enemyHealth <= 0) {
                        handleVictory();
                    } else {
                        currentQuestionIndex++;
                        loadQuestion();
                        if (button) button.classList.remove('correct');
                    }
                });
            });

            gradualHealthDecrease(enemyHealthBar, enemyHpCounter, enemyHealth, enemyHealth - 20);
            enemyHealth -= 20;

        } else {
            if (button) button.classList.add('incorrect');
            console.log("Incorrect answer or timeout, initiating enemy run and attack animations.");

            playEnemyRunAnimation(function() {
                playEnemyAttackAnimation(function() {
                    if (playerHealth <= 0) {
                        handleDefeat();
                    } else {
                        playIdleAnimation();
                        if (button) button.classList.remove('incorrect');
                        enableAnswerButtons();
                        currentQuestionIndex++;
                        loadQuestion();
                    }
                });
            });

            gradualHealthDecrease(playerHealthBar, playerHpCounter, playerHealth, playerHealth - 20);
            playerHealth -= 20;
        }
    }

    // Function to load the question
    function loadQuestion() {
        if (currentQuestionIndex < stageQuestions.length) {
            const questionData = stageQuestions[currentQuestionIndex];
            questionText.textContent = questionData.question;

            answerButtonsContainer.innerHTML = ''; // Clear previous buttons
            questionData.options.forEach((option, index) => {
                const button = document.createElement('button');
                button.className = 'answer-button';
                button.textContent = option;
                button.setAttribute('data-correct', index === questionData.correct_option);
                button.addEventListener('click', function() {
                    const isCorrect = button.getAttribute('data-correct') === 'true';
                    handleAnswer(button, isCorrect, bossName);
                });
                answerButtonsContainer.appendChild(button);
            });

            // Start the answer timer for this question
            startAnswerTimer();
        }
    }

    fetch('static/json/questions.json')
        .then(response => response.json())
        .then(data => {
            stageQuestions = data[`stage${stage}`].questions;
            shuffle(stageQuestions); // Shuffle questions to randomize order
            loadQuestion(); // Load the first question on page load
        })
        .catch(error => {
            console.error('Error loading questions:', error);
        });

    function updateHealthBar(healthBar, health) {
        healthBar.style.width = `${Math.max(health, 0)}%`;
    }

    function updateHpCounter(hpCounter, currentHp, maxHp) {
        hpCounter.textContent = `${Math.max(currentHp, 0)}/${maxHp} HP`;
    }

    function gradualHealthDecrease(healthBar, hpCounter, currentHealth, targetHealth) {
        const decreaseRate = 1; // How much health to decrease per frame
        const interval = setInterval(() => {
            if (currentHealth > targetHealth) {
                currentHealth -= decreaseRate;
                updateHealthBar(healthBar, currentHealth);
                updateHpCounter(hpCounter, currentHealth, maxHealth);
            } else {
                clearInterval(interval);
            }
        }, 1000 / 60); // Adjusted interval to match 5 FPS
    }

    function handleVictory() {
        if (stage === 5) {
            // Redirect to /endgame if the user beats the last boss
            window.location.href = '/endgame';
        } else {
            // Show the victory screen and proceed to the next stage
            showGameOverScreen(`You defeated ${bossName}!`, 'Mission Status: Success', "You've advanced to the next stage.", 'Next Stage', true);
            unlockNextStage(stage);
        }
    }
    
    function handleDefeat() {
        showGameOverScreen(`You were defeated by ${bossName}!`, 'Mission Status: Failure', "Try again to overcome the challenge.", 'Retry', false);
    }

    function showGameOverScreen(title, summary, message, buttonText, isWin) {
        gameOverTitle.textContent = title;
        gameOverSummary.textContent = summary;
        gameOverMessage.textContent = message;
        gameOverButtons.innerHTML = `<button onclick="window.location.href='/stages'">${buttonText}</button>`;
        document.querySelector('.fight-container').style.display = 'none'; // Hide the fight elements
        gameOverScreen.style.display = 'flex'; // Show the game over screen
        
        if (isWin) {
            gameOverTitle.style.color = '#00ff00'; // Green color for win
        } else {
            gameOverTitle.style.color = '#ff0000'; // Red color for loss
        }
    }

    function unlockNextStage(currentStage) {
        let unlockedStages = localStorage.getItem('unlockedStages');
        unlockedStages = unlockedStages ? JSON.parse(unlockedStages) : [1];

        const nextStage = currentStage + 1;

        updateStageStatus(currentStage, 'Completed');

        if (nextStage <= 5 && !unlockedStages.includes(nextStage)) {
            unlockedStages.push(nextStage);
            localStorage.setItem('unlockedStages', JSON.stringify(unlockedStages));
            updateStageStatus(nextStage, 'Ongoing');
            console.log(`Stage ${nextStage} unlocked and set to Ongoing!`);
        } else {
            console.log(`Stage ${nextStage} is either already unlocked or doesn't exist.`);
        }
    }

    function updateStageStatus(stageNumber, status) {
        let stageStatus = localStorage.getItem('stageStatus') ? JSON.parse(localStorage.getItem('stageStatus')) : {};
        stageStatus[stageNumber] = status;
        localStorage.setItem('stageStatus', JSON.stringify(stageStatus));
    }

    // Helper function to shuffle an array
    function shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }
});

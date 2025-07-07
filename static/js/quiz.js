// Quiz functionality - Updated for new design
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

function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

function startQuiz() {
    // Hide start screen and results
    document.getElementById('quiz-start-screen').style.display = 'none';
    document.getElementById('results-container').style.display = 'none';
    
    // Show quiz container and progress
    document.getElementById('quiz-container').style.display = 'block';
    document.getElementById('progress-container').style.display = 'block';
    
    // Reset quiz state
    score = 0;
    currentQuestionIndex = 0;

    // Shuffle questions and select 5
    shuffle(questions);
    currentQuestions = questions.slice(0, 5);

    // Reset progress bar
    updateProgress();
    
    // Display first question
    displayQuestion();
}

function updateProgress() {
    const progressFill = document.getElementById('progress-fill');
    const currentQuestionSpan = document.getElementById('current-question');
    const progressPercentage = document.getElementById('progress-percentage');
    
    const progress = ((currentQuestionIndex) / currentQuestions.length) * 100;
    
    progressFill.style.width = progress + '%';
    currentQuestionSpan.textContent = currentQuestionIndex + 1;
    progressPercentage.textContent = Math.round(progress) + '%';
}

function displayQuestion() {
    if (currentQuestionIndex >= currentQuestions.length) {
        showResults();
        return;
    }

    const quizContainer = document.getElementById('quiz-container');
    quizContainer.innerHTML = '';

    const questionObj = currentQuestions[currentQuestionIndex];
    
    // Create question element
    const questionElem = document.createElement('div');
    questionElem.classList.add('quiz-question');
    questionElem.textContent = `${questionObj.question}`;

    // Create options list
    const optionsList = document.createElement('ul');
    optionsList.classList.add('quiz-options');

    questionObj.options.forEach((option, index) => {
        const optionElem = document.createElement('li');
        optionElem.textContent = option;
        optionElem.onclick = () => checkAnswer(optionElem, index);
        optionsList.appendChild(optionElem);
    });

    quizContainer.appendChild(questionElem);
    quizContainer.appendChild(optionsList);

    // Create explanation placeholder
    const explanationElem = document.createElement('div');
    explanationElem.classList.add('explanation');
    explanationElem.style.display = 'none';
    quizContainer.appendChild(explanationElem);

    // Create next button
    const nextButton = document.createElement('button');
    nextButton.textContent = currentQuestionIndex < currentQuestions.length - 1 ? 'Next Question' : 'Finish Quiz';
    nextButton.classList.add('next-button');
    nextButton.style.display = 'none';
    nextButton.onclick = () => {
        currentQuestionIndex++;
        updateProgress();
        displayQuestion();
    };
    quizContainer.appendChild(nextButton);
}

function checkAnswer(optionElem, selectedIndex) {
    const correctAnswer = currentQuestions[currentQuestionIndex].answer;
    const explanationElem = document.querySelector('.explanation');
    const nextButton = document.querySelector('.next-button');
    
    // Remove previous selections
    const options = document.querySelectorAll('.quiz-options li');
    options.forEach(option => {
        option.onclick = null; // Disable all options
        option.classList.remove('selected', 'correct', 'incorrect');
    });

    // Mark selected option
    optionElem.classList.add('selected');
    
    if (selectedIndex === correctAnswer) {
        optionElem.classList.add('correct');
        explanationElem.innerHTML = `<strong>Correct!</strong> ${currentQuestions[currentQuestionIndex].explanation}`;
        score++;
    } else {
        optionElem.classList.add('incorrect');
        // Also highlight the correct answer
        options[correctAnswer].classList.add('correct');
        explanationElem.innerHTML = `<strong>Incorrect.</strong> ${currentQuestions[currentQuestionIndex].explanation}`;
    }

    // Show explanation and next button
    explanationElem.style.display = 'block';
    nextButton.style.display = 'inline-block';
}

function showResults() {
    // Hide quiz container and progress
    document.getElementById('quiz-container').style.display = 'none';
    document.getElementById('progress-container').style.display = 'none';
    
    // Show results
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.style.display = 'block';
    
    // Update results content
    const scoreDisplay = document.getElementById('score');
    const messageDisplay = document.getElementById('result-message');
    const resultsIcon = document.getElementById('results-icon');

    scoreDisplay.textContent = score;

    // Set message and icon based on score
    if (score === 5) {
        messageDisplay.textContent = "Perfect score! You're a deepfake detection expert! 🎉";
        messageDisplay.className = "result-message perfect";
        resultsIcon.textContent = "🏆";
        
        // Trigger confetti for perfect score
        if (typeof confetti !== 'undefined') {
            confetti({
                particleCount: 100,
                spread: 70,
                origin: { y: 0.6 }
            });
        }
    } else if (score >= 4) {
        messageDisplay.textContent = "Excellent work! You have a strong understanding of deepfake technology.";
        messageDisplay.className = "result-message good";
        resultsIcon.textContent = "🎯";
    } else if (score >= 3) {
        messageDisplay.textContent = "Good job! You're on the right track. Keep learning!";
        messageDisplay.className = "result-message good";
        resultsIcon.textContent = "👍";
    } else {
        messageDisplay.textContent = "Keep studying! Every expert started as a beginner.";
        messageDisplay.className = "result-message try-again";
        resultsIcon.textContent = "📚";
    }
}

// Auto-start functionality
document.addEventListener('DOMContentLoaded', function() {
    // Quiz starts with the start screen visible by default
    // No auto-start needed - user clicks the "Begin Quiz" button
});
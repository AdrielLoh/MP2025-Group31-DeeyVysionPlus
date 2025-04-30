// Define different endings based on player performance
const endings = {
    perfect: [
        "Incredible, Agent DV!",
        "You completed the mission without a scratch. Not a single error in your flawless run.",
        "Your performance was beyond exceptional, earning you the title of Ultimate Defender of the Digital Realm!",
        "The world is in awe of your skills. Thank you for playing!"
    ],
    good: [
        "Well done, Agent DV!",
        "You successfully defeated all enemies and restored peace to the digital realm.",
        "Your journey was filled with challenges, but you overcame them with determination and skill.",
        "You are awarded the title of Master Defender of the Digital Realm. The world is safer because of you."
    ],
    bad: [
        "Agent DV, you made it through, but not without some trouble.",
        "While you completed the mission, your journey was fraught with difficulties.",
        "You may need to hone your skills further. Nonetheless, you are recognized as a Defender of the Digital Realm.",
        "Thank you for playing. Consider replaying to improve your performance!"
    ]
};

// Placeholder: Determine player's performance (this would be based on your game logic)
let playerPerformance = 'good'; // This could be 'perfect', 'good', or 'bad' based on the player's actual performance

const endgameDialogues = endings[playerPerformance]; // Use the appropriate ending dialogues

let endgameDialogueIndex = 0;
let typingSpeed = 30; // Adjust typing speed (lower is faster)
let typingTimeout; // Store the timeout for the typing effect
let isTyping = false; // Track whether typing is in progress

function nextEndgameDialogue() {
    if (isTyping) return;

    const currentDialogue = endgameDialogues[endgameDialogueIndex];
    document.getElementById('endgame-dialogue-text').innerText = ''; // Clear text before typing

    typeDialogue(currentDialogue, 0); // Start typing

    if (endgameDialogueIndex < endgameDialogues.length - 1) {
        endgameDialogueIndex++;
    } else {
        document.getElementById('endgame-next-button').style.display = 'none';
        document.querySelector('.endgame-buttons').style.display = 'block';
    }
}

function typeDialogue(text, i) {
    if (i < text.length) {
        isTyping = true;
        document.getElementById('endgame-dialogue-text').innerText += text.charAt(i);
        typingTimeout = setTimeout(() => typeDialogue(text, i + 1), typingSpeed);
    } else {
        isTyping = false;
    }
}

function replayGame() {
    localStorage.clear();
    window.location.reload();
    window.location.href = '/stages';
}

function backToStages() {
    window.location.href = '/stages';
}

// Initialize endgame page
window.onload = () => {
    document.querySelector('.endgame-buttons').style.display = 'none'; // Hide buttons initially
    nextEndgameDialogue(); // Start the first dialogue automatically
};

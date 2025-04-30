let dialogueIndex = 0;
let typingSpeed = 30;
let typingTimeout;
let isTyping = false;

const stage = new URLSearchParams(window.location.search).get('stage');

const dialogues = {
    1: [
        "Welcome, Agent DV. Your mission is about to begin...",
        "You are Agent DV, a digital vigilante created to protect the virtual world from malicious entities.",
        "In a time where deepfakes and digital deceptions threaten to disrupt societies, your role is critical.",
        "Your first opponent is the AI Overlord. A formidable foe who can outsmart any machine.",
        "The AI Overlord has mastered the art of generating deepfakes that are indistinguishable from reality.",
        "Prepare yourself, Agent DV. The AI Overlord awaits you in the battle arena."
    ],
    2: [
        "Agent DV, you’ve done well so far. But now, you face the Fabricator Phantom.",
        "The Fabricator Phantom is a rogue AI, designed to fabricate elaborate lies and illusions.",
        "This master of deception weaves falsehoods that have caused havoc across digital landscapes.",
        "It’s your task to expose the truth and dismantle the Fabricator Phantom’s network of deceit.",
        "Enter the battlefield, and expose the Fabricator Phantom for what it truly is."
    ],
    3: [
        "Your next challenge is the Misinformation Master, Agent DV.",
        "This insidious foe thrives on spreading falsehoods and misinformation, causing chaos and confusion.",
        "The Misinformation Master has weaponized fake news, manipulating the masses for its gain.",
        "As Agent DV, you must cut through the lies and bring clarity back to the digital world.",
        "Get ready to confront the Misinformation Master in the arena."
    ],
    4: [
        "Agent DV, the Identity Shifter awaits. This enemy can change forms at will.",
        "The Identity Shifter is an elusive adversary, capable of stealing and mimicking identities with precision.",
        "It has misled countless victims by disguising itself, leaving trails of devastation.",
        "Your purpose is to unmask this foe and restore the integrity of identities in the virtual realm.",
        "Enter the battlefield, and reveal the Identity Shifter’s true form."
    ],
    5: [
        "Your final challenge is here, Agent DV. The Reality Warper threatens all we hold dear.",
        "The Reality Warper is the most dangerous of all. It bends reality itself, distorting truths and creating alternate false realities.",
        "This adversary seeks to collapse the digital and physical worlds into chaos, erasing the line between truth and fiction.",
        "As the last line of defense, your mission is to destroy the Reality Warper and safeguard the future.",
        "Prepare for the final showdown with the Reality Warper. The fate of both worlds rests in your hands."
    ]
};

function nextDialogue() {
    if (isTyping) return;

    const currentDialogue = dialogues[stage][dialogueIndex];
    document.getElementById('dialogue-text').innerText = '';

    typeDialogue(currentDialogue, 0);

    if (dialogueIndex < dialogues[stage].length - 1) {
        dialogueIndex++;
    } else {
        // Hide the "Next" button and show the "Start Fight" button
        document.getElementById('next-button').style.display = 'none';
        document.getElementById('skip-button').style.display = 'none';
        document.getElementById('start-button').style.display = 'block'; // Show the Start Fight button
    }
}

function typeDialogue(text, i) {
    if (i < text.length) {
        isTyping = true;
        document.getElementById('dialogue-text').innerText += text.charAt(i);
        typingTimeout = setTimeout(() => typeDialogue(text, i + 1), typingSpeed);
    } else {
        isTyping = false;
    }
}

function skipDialogue() {
    clearTimeout(typingTimeout); // Stop the current typing animation
    dialogueIndex = dialogues[stage].length - 1; // Move to the last dialogue
    document.getElementById('dialogue-text').innerText = dialogues[stage][dialogueIndex]; // Set the last dialogue text instantly
    
    // Hide the "Next" and "Skip" buttons and show the "Start Fight" button
    document.getElementById('next-button').style.display = 'none';
    document.getElementById('skip-button').style.display = 'none';
    document.getElementById('start-button').style.display = 'block';
    
    isTyping = false; // Ensure that the typing effect is stopped
}


// Start the fight
function startFight() {
    window.location.href = `/fight?stage=${stage}`;
}

// Initialize the dialogue on page load
window.onload = () => {
    dialogueIndex = 0;
    document.getElementById('start-button').style.display = 'none'; // Hide Start Fight button on load
    nextDialogue(); 
};

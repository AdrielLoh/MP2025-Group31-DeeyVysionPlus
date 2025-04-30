document.addEventListener('DOMContentLoaded', (event) => {
    
    checkUnlockedStages();
    document.getElementById('reset-progress-button').addEventListener('click', resetProgress);
});

function checkUnlockedStages() {
    const unlockedStages = getUnlockedStages();
    let stageStatus = localStorage.getItem('stageStatus') ? JSON.parse(localStorage.getItem('stageStatus')) : {};

    for (let i = 1; i <= 5; i++) {
        const stageElement = document.getElementById(`stage${i}`);
        if (unlockedStages.includes(i)) {
            stageElement.classList.remove('locked');
            stageElement.classList.add('unlocked');
            stageElement.onclick = () => selectStage(i);

            // Update the status based on stored data
            const status = stageStatus[i] || 'Not Completed';
            updateStageStatus(i, status);
        } else {
            stageElement.classList.remove('unlocked');
            stageElement.classList.add('locked');
            stageElement.onclick = null;
        }
    }
}

function getUnlockedStages() {
    let unlockedStages = localStorage.getItem('unlockedStages');
    if (!unlockedStages) {
        unlockedStages = [1]; // Default to Stage 1 being unlocked
        localStorage.setItem('unlockedStages', JSON.stringify(unlockedStages));
    } else {
        unlockedStages = JSON.parse(unlockedStages);
    }
    return unlockedStages;
}

function selectStage(stageNumber) {
    const stageElement = document.getElementById(`stage${stageNumber}`);
    const statusElement = stageElement.querySelector('.status').textContent;

    if (statusElement === 'Completed') {
        showModal();
    } else if (stageElement.classList.contains('unlocked')) {
        window.location.href = `/dialogues?stage=${stageNumber}`; // Navigate to dialogues with the correct stage number
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
        console.log(`Stage ${nextStage} unlocked!`);
    } else {
        console.log(`Stage ${nextStage} is either already unlocked or doesn't exist.`);
    }
}

function updateStageStatus(stageNumber, status) {
    const stageElement = document.getElementById(`stage${stageNumber}`);
    
    if (!stageElement) {
        console.error(`Stage element with ID stage${stageNumber} not found.`);
        return;
    }
    
    const statusElement = stageElement.querySelector('.status');
    
    if (statusElement) {
        statusElement.textContent = status;
        if (status === 'Completed') {
            statusElement.classList.remove('ongoing');
            statusElement.classList.add('completed');
        } else if (status === 'Ongoing') {
            statusElement.classList.remove('completed');
            statusElement.classList.add('ongoing');
        }
    } else {
        console.error(`Status element not found for stage ${stageNumber}.`);
    }
}

// Modal handling
function showModal() {
    const modal = document.getElementById('completed-stage-modal');
    modal.classList.add('show');
}

function closeModal() {
    const modal = document.getElementById('completed-stage-modal');
    modal.classList.remove('show');
}

document.querySelector('.close-button').addEventListener('click', closeModal);
document.querySelector('.modal-close-button').addEventListener('click', closeModal);
window.addEventListener('click', function(event) {
    const modal = document.getElementById('completed-stage-modal');
    if (event.target === modal) {
        closeModal();
    }
});

function resetProgress() {
    localStorage.clear();
    window.location.reload(); // Reload the page to reflect the reset state
}

// Ensure these variables are declared only once and uniquely named
const animationCanvas = document.getElementById('animationCanvas');
const ctx = animationCanvas.getContext('2d');

const spriteSheet = new Image();
spriteSheet.src = 'static/images/game/sprites/agent-dv.png';

let animationData; // Declare animationData globally

// Load the JSON data
fetch('static/json/agent-dv.json')
    .then(response => response.json())
    .then(data => {
        animationData = processAnimationData(data.frames);
        
        // Example: Play the attack animation
        playAnimation('attack', 10);
    })
    .catch(error => console.error('Error loading JSON:', error));

function processAnimationData(frames) {
    const animationData = {};

    // Loop through each frame in the JSON
    for (const [fileName, frameData] of Object.entries(frames)) {
        // Extract the animation type from the filename
        const animationType = fileName.split('_')[2]; // e.g., "attack", "idle", "run"

        // Initialize the array if it doesn't exist
        if (!animationData[animationType]) {
            animationData[animationType] = [];
        }

        // Add the frame to the correct animation type
        animationData[animationType].push({
            frame: parseFrame(frameData.frame),
            offset: parseCoordinate(frameData.offset),
            rotated: frameData.rotated,
            sourceColorRect: parseFrame(frameData.sourceColorRect),
            sourceSize: parseCoordinate(frameData.sourceSize)
        });
    }

    return animationData;
}

function parseFrame(frameString) {
    const [pos, size] = frameString.replace(/{{|}}/g, '').split('},{');
    const [x, y] = pos.split(',').map(Number);
    const [width, height] = size.split(',').map(Number);
    return { x, y, width, height };
}

function parseCoordinate(coordinateString) {
    return coordinateString.replace(/{|}/g, '').split(',').map(Number);
}

function playAnimation(type, frameRate = 10) {
    if (spriteSheet.complete && spriteSheet.naturalWidth !== 0) {
        const frames = animationData[type];
        let index = 0;
        const delay = 1000 / frameRate;

        function animate() {
            const { x, y, width, height } = frames[index].frame;

            ctx.clearRect(0, 0, animationCanvas.width, animationCanvas.height);
            ctx.drawImage(spriteSheet, x, y, width, height, 0, 0, animationCanvas.width, animationCanvas.height);

            index = (index + 1) % frames.length;

            if (index !== 0) {
                setTimeout(animate, delay);
            }
        }

        animate();
    } else {
        console.error('Sprite sheet not fully loaded. Unable to play animation.');
    }
}

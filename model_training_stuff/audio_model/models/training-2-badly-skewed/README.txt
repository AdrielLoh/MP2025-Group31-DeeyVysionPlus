This model was trained on lightly augmented audio for both real and fake samples.

The scaler was fitted using non-augmented audio, which I suspect might be the cause for the following issue:
	- Validation and testing were done on non-augmented audio, but there is a major skew towards fake predictions.
	- Around 60% of real testing samples were falsely classified as fake, while 99% of fake samples were correctly classified as fake.

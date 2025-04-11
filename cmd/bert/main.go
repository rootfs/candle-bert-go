package main

import (
	"fmt"
	"os"

	"github.com/candle-bert-go/pkg/classifier"
)

func main() {
	// Initialize the BERT model with 3 classes
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "sentence-transformers/all-MiniLM-L6-v2"
	}

	numClasses := 3 // Define 3 classes
	useCPU := true

	success := classifier.InitBert(modelPath, numClasses, useCPU)
	if !success {
		fmt.Printf("Failed to initialize BERT\n")
		return
	}

	testTexts := []string{
		// Simple texts
		"Hello world",
		"The cat sat on the mat.",
		"I like to read books.",

		// Moderate complexity
		"Machine learning algorithms can be categorized as supervised or unsupervised, with each approach having distinct advantages in different contexts.",
		"Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities, especially the burning of fossil fuels.",
		"The economic impact of the pandemic has highlighted structural inequalities in healthcare access and employment stability across different sectors.",

		// Complex texts
		"Quantum entanglement is a physical phenomenon that occurs when a pair or group of particles is generated, interact, or share spatial proximity in a way such that the quantum state of each particle of the pair or group cannot be described independently of the state of the others, including when the particles are separated by a large distance.",
		"The integration of General Relativity with Quantum Mechanics remains one of the most significant challenges in theoretical physics, with approaches like String Theory and Loop Quantum Gravity attempting to reconcile the mathematical inconsistencies between these fundamental frameworks.",
		"Metacognitive regulation encompasses planning (selecting appropriate strategies and allocating resources), monitoring (awareness of understanding and task performance), and evaluating (appraising the final product of a task and the efficiency at which the task was performed) one's cognition.",
	}

	fmt.Println("\n======= Classifying Texts =======\n")

	for i, text := range testTexts {
		// Classify text using our library with confidence information
		result := classifier.ClassifyTextWithConfidence(text)

		if result.Class < 0 {
			fmt.Printf("Error classifying text #%d\n", i+1)
			continue
		}

		// Show sample number and truncated text if too long
		displayText := text
		if len(displayText) > 100 {
			displayText = displayText[:97] + "..."
		}

		fmt.Printf("Text #%d: '%s'\n", i+1, displayText)
		fmt.Printf("Classification: %s with confidence: %.2f%%\n",
			result.ClassName, result.Confidence*100)
	}
}

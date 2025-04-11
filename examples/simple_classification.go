package main

import (
	"fmt"
	"os"

	"github.com/candle-bert-go/pkg/classifier"
)

func main() {
	// Initialize the BERT model
	modelPath := os.Getenv("MODEL_PATH")
	if modelPath == "" {
		modelPath = "sentence-transformers/all-MiniLM-L6-v2"
	}

	numClasses := 3
	useCPU := true

	fmt.Println("Initializing BERT model with", numClasses, "classes...")
	success := classifier.InitBert(modelPath, numClasses, useCPU)
	if !success {
		fmt.Println("Failed to initialize BERT model")
		os.Exit(1)
	}

	// Simple example texts
	texts := []string{
		"This is a very simple sentence.",
		"Machine learning involves using algorithms to learn patterns from data.",
		"Quantum computing leverages quantum phenomena such as superposition and entanglement to process information in ways that classical computers cannot, potentially leading to exponential speedups for certain types of problems.",
	}

	// Classify each text with confidence scores
	fmt.Println("\n=== Text Classification Results with Confidence ===\n")
	for i, text := range texts {
		// Get classification with confidence
		result := classifier.ClassifyTextWithConfidence(text)

		if result.Class < 0 {
			fmt.Printf("Error classifying text #%d\n", i+1)
			continue
		}

		fmt.Printf("Text: '%s'\n", text)
		fmt.Printf("Classification: %s with confidence: %.2f%%\n",
			result.ClassName, result.Confidence*100)
	}
}

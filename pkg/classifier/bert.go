package classifier

import (
	"fmt"
	"unsafe"
)

/*
#cgo LDFLAGS: -L../../rust/target/release -lcandle_bert_go -ldl -lm
#include <stdlib.h>
#include <stdbool.h>

extern bool init_bert(const char* model_id, int num_classes_or_use_cpu, bool use_cpu);
extern int classify_text(const char* text);

// Classification result structure
typedef struct {
    int class;
    float confidence;
} ClassificationResult;

extern ClassificationResult classify_text_with_confidence(const char* text);
extern void free_cstring(char* s);
*/
import "C"

// ClassNames for the classifier
var ClassNames = []string{"class 0", "class 1", "class 2"}

// Classification result with class index and confidence score
type Classification struct {
	Class      int
	Confidence float32
	ClassName  string
}

// InitBert initializes the BERT model with the specified model path and number of classes
func InitBert(modelPath string, numClasses int, useCPU bool) bool {
	if modelPath == "" {
		modelPath = "sentence-transformers/all-MiniLM-L6-v2"
	}

	fmt.Println("Initializing BERT model:", modelPath, "with", numClasses, "classes")

	// Initialize BERT directly using CGO
	cModelID := C.CString(modelPath)
	defer C.free(unsafe.Pointer(cModelID))

	success := C.init_bert(cModelID, C.int(numClasses), C.bool(useCPU))
	return bool(success)
}

// ClassifyText classifies the input text and returns the class index only (backward compatibility)
func ClassifyText(text string) int {
	cText := C.CString(text)
	result := C.classify_text(cText)
	C.free(unsafe.Pointer(cText))

	return int(result)
}

// ClassifyTextWithConfidence classifies the input text and returns both class index and confidence
func ClassifyTextWithConfidence(text string) Classification {
	cText := C.CString(text)
	result := C.classify_text_with_confidence(cText)
	C.free(unsafe.Pointer(cText))

	classIdx := int(result.class)
	confidence := float32(result.confidence)
	className := GetClassName(classIdx)

	return Classification{
		Class:      classIdx,
		Confidence: confidence,
		ClassName:  className,
	}
}

// GetClassName returns the class name for the given index
func GetClassName(classIdx int) string {
	if classIdx >= 0 && classIdx < len(ClassNames) {
		return ClassNames[classIdx]
	}
	return "unknown"
}

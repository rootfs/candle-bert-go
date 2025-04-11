# Candle BERT Go

Go bindings for the Candle machine learning framework to run BERT models with text classification capabilities.

## Overview

This project provides Go bindings for the Candle machine learning framework, allowing you to:

1. Load pre-trained BERT models
2. Classify text into multiple categories
3. Integrate into Go applications

## Project Structure

```
candle-bert-go/
├── cmd/               # Go command-line applications
│   └── bert/          # BERT text classification CLI
├── examples/          # Example Go applications
│   └── simple_classification.go  # Simple classification example
├── pkg/               # Go packages
│   └── classifier/    # Text classification library
├── rust/              # Rust code for Candle bindings
│   ├── src/           # Rust source files
│   └── Cargo.toml     # Rust dependencies
├── scripts/           # Build and utility scripts
├── Makefile           # Build automation
├── .gitignore         # Git ignore rules
└── go.mod             # Go module definition
```

## Prerequisites

- Go 1.21+
- Rust and Cargo

## Building

You can use the provided Makefile for easy building:

```bash
# Build both Rust and Go components
make

# Build only the Rust library
make rust

# Build only the Go application  
make go

# Build example applications
make examples
```

## Running

Using the Makefile:

```bash
# Run the main application
make run

# Run the simple classification example
make run-example
```

## Using as a Library

You can use this package as a library in your own Go applications:

```go
import "github.com/rootfs/candle-bert-go/pkg/classifier"

// Initialize the BERT model
success := classifier.InitBert("sentence-transformers/all-MiniLM-L6-v2", 3, true)

// Classify text
classIdx := classifier.ClassifyText("Your text to classify")

// Get readable class name
className := classifier.GetClassName(classIdx)
```

See the `examples/` directory for more examples.

## Cleaning Up

```bash
make clean
```

## Additional Makefile Commands

- `make test` - Run tests
- `make install` - Install to GOPATH/bin
- `make docs` - Generate documentation
- `make help` - Show help information

## License

MIT or Apache 2.0 license.
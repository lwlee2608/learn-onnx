package main

import (
	"fmt"
	ort "github.com/yalue/onnxruntime_go"
	// "os"
)

func main() {
	// This line _may_ be optional; by default the library will try to load
	// "onnxruntime.dll" on Windows, and "onnxruntime.so" on any other system.
	// For stability, it is probably a good idea to always set this explicitly.
	lib := "/usr/local/lib/onnxruntime/lib/libonnxruntime.so"
	ort.SetSharedLibraryPath(lib)

	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	defer ort.DestroyEnvironment()

	// For a slight performance boost and convenience when re-using existing
	// tensors, this library expects the user to create all input and output
	// tensors prior to creating the session. If this isn't ideal for your use
	// case, see the DynamicAdvancedSession type in the documnentation, which
	// allows input and output tensors to be specified when calling Run()
	// rather than when initializing a session.
	inputData := []float32{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
	inputShape := ort.NewShape(2, 5)
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	defer inputTensor.Destroy()
	// This hypothetical network maps a 2x5 input -> 2x3x4 output.
	outputShape := ort.NewShape(2, 3, 4)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	defer outputTensor.Destroy()

	model := "jina/model/model.onnx"
	session, err := ort.NewAdvancedSession(model,
		[]string{"Input 1 Name"}, []string{"Output 1 Name"},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
	defer session.Destroy()

	// Calling Run() will run the network, reading the current contents of the
	// input tensors and modifying the contents of the output tensors.
	err = session.Run()

	// Get a slice view of the output tensor's data.
	outputData := outputTensor.GetData()

	fmt.Println("Output data:", outputData)

	// If you want to run the network on a different input, all you need to do
	// is modify the input tensor data (available via inputTensor.GetData())
	// and call Run() again.

	// ...
}

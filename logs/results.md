# Results raw logs

## CPU - no optimization
Model: HuggingFaceTB/SmolLM-135M
Strategy: none
Backend: base_cpu
Dataset: squad (validation)
Samples processed: 50
Total tokens generated: 2500
Total energy consumed: 0.003322 kWh
Total duration: 253.09974167700102 seconds
Energy per token: 0.0013287840479939944 Wh/token
Duration per token: 0.1012 seconds/token
CPU energy per token: 0.0011951444824550063 Wh/token
GPU energy per token: 0.0 Wh/token
RAM energy per token: 0.0001336395655389882 Wh/token
Equivalent CO2 emissions per token: 4.6404621535691e-07 kg eqCO2/token

## CPU - dynamic 16-bit quantization
Model: HuggingFaceTB/SmolLM-135M
Strategy: none
Backend: onnx_cpu
Dataset: squad (validation)
Samples processed: 50
Total tokens generated: 2500
Total energy consumed: 0.002284 kWh
Total duration: 144.35228206200003 seconds
Energy per token: 0.0009136028977141554 Wh/token
Duration per token: 0.0577 seconds/token
CPU energy per token: 0.0006808806735344435 Wh/token
GPU energy per token: 0.00015659112527280003 Wh/token
RAM energy per token: 7.61310989069119e-05 Wh/token
Equivalent CO2 emissions per token: 3.1905407629131623e-07 kg eqCO2/token

## GPU - static 8-bit quantization
Model: HuggingFaceTB/SmolLM-135M
Strategy: quantization
Backend: onnx_cpu
Dataset: squad (validation)
Samples processed: 50
Total tokens generated: 2065
Total energy consumed: 0.001149 kWh
Total duration: 72.99968414700027 seconds
Energy per token: 0.0005562886868804103 Wh/token
Duration per token: 0.0354 seconds/token
CPU energy per token: 0.0004163742874697345 Wh/token
GPU energy per token: 9.337645350121044e-05 Wh/token
RAM energy per token: 4.653794590946566e-05 Wh/token
Equivalent CO2 emissions per token: 1.9427058910168842e-07 kg eqCO2/token

## GPU - no optimization
Model: HuggingFaceTB/SmolLM-135M
Strategy: none
Backend: base_gpu
Dataset: squad (validation)
Samples processed: 50
Total tokens generated: 2500
Total energy consumed: 0.001734 kWh
Total duration: 81.44930677199955 seconds
Energy per token: 0.0006935782046561975 Wh/token
Duration per token: 0.0326 seconds/token
CPU energy per token: 0.00038403479175222265 Wh/token
GPU energy per token: 0.00026660921328720106 Wh/token
RAM energy per token: 4.293419961677384e-05 Wh/token
Equivalent CO2 emissions per token: 2.422156868985858e-07 kg eqCO2/token

## GPU - static 16-bit quantization
Model: HuggingFaceTB/SmolLM-135M
Strategy: quantization16bit
Backend: base_gpu
Dataset: squad (validation)
Samples processed: 50
Total tokens generated: 2500
Total energy consumed: 0.002569 kWh
Total duration: 119.6750711790005 seconds
Energy per token: 0.0010277827542389064 Wh/token
Duration per token: 0.0479 seconds/token
CPU energy per token: 0.0005645254679097256 Wh/token
GPU energy per token: 0.0004001372089984006 Wh/token
RAM energy per token: 6.312007733078046e-05 Wh/token
Equivalent CO2 emissions per token: 3.5892867470352196e-07 kg eqCO2/token

## GPU - static 8-bit quantization
Model: HuggingFaceTB/SmolLM-135M
Strategy: quantization8bit
Backend: base_gpu
Dataset: squad (validation)
Samples processed: 50
Total tokens generated: 2500
Total energy consumed: 0.005477 kWh
Total duration: 252.8325312540003 seconds
Energy per token: 0.0021908850836827062 Wh/token
Duration per token: 0.1011 seconds/token
CPU energy per token: 0.0011933211162599982 Wh/token
GPU energy per token: 0.0008641225801864002 Wh/token
RAM energy per token: 0.00013344138723630862 Wh/token
Equivalent CO2 emissions per token: 7.65114491628411e-07 kg eqCO2/token

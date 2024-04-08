This repo contains the simulation codes presented in the paper titled [Federated Learning Under a Digital Communication Model](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10363220)

This project is licensed under the [Apache-2.0 license] - see the [LICENSE](LICENSE) file for details.

Abstract:
Federated learning (FL) has received significant attention recently as a topic in distributed learning. In FL, a global model is cooperatively trained by edge devices, as agents, where data is locally generated, processed, and utilized. Agents communicate with a server once in a while after local updates. Since many agents may participate in FL, the underlying communication is typically considered analog, with over-the-air aggregation at the server to alleviate the communication bottleneck. However, digital communication offers favorable programmability and additional digital processing capabilities such as compression and encryption, motivating its use in FL. To mitigate communication bandwidth requirements, existing digital communication models for FL employ over-the-air aggregation of modulated symbols, where limited quantization levels and small constellation sizes are required. Motivated by this limitation, this paper investigates FL under an error-prone digital communication model to leverage the existing infrastructure more efficiently. Based on the established analyses, convergence is guaranteed as long as the bias in expected local stochastic gradients, arising from quantization and transmission errors, diminishes. According to this result, the considered communication model employs a digital modulation scheme, enabling communication to be error-prone in the first iterations, prioritizing limited bandwidth, power consumption, and speed. In later iterations, it has the option to decrease the bit error rate (BER) to guarantee convergence to a neighborhood of some stationary point. Numerical experiments on classification tasks reveal promising generalization, possible convergence speedup in the initial iterations, and communication bandwidth, latency, and power efficiency.

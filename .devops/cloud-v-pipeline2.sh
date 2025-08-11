node{  // Use any available node
    stage('Cleanup'){
        cleanWs()               // Cleaning previous CI build in workspace
    }
    stage('checkout repo'){
        retry(5){               // Retry if the cloning fails due to some reason
            git url: 'https://github.com/akeana-sw/llama.cpp.git', branch: 'master'
        }
    }
    stage('Compiling llama.cpp'){
        sh'''#!/bin/bash
            mkdir -p build
            cd build
            cmake .. \
                -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc \
                -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++ \
                -DCMAKE_SYSTEM_NAME=Linux \
                -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
                -DGGML_RVV=OFF \
                -DLLAMA_BUILD_TESTS=OFF \
                -DLLAMA_BUILD_EXAMPLES=OFF \
                -DCMAKE_CXX_FLAGS="-fpermissive" \
                -DLLAMA_BUILD_SERVER=OFF \
                -DLLAMA_CURL=OFF
            make -j$(nproc) llama-cli
        '''
    }
    stage('Running llama.cpp'){
        sh'''#!/bin/bash
            MODEL_PATH="/proj/local/public_models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
            QEMU_CMD="/proj/sw_tools/ml_qemu/build/qemu-riscv64"
            
            ${QEMU_CMD} -L /usr/riscv64-linux-gnu \
                -cpu rv64,v=true,vlen=256,elen=64,vext_spec=v1.0 \
                ./build/bin/llama-cli -m ${MODEL_PATH} -p "Hello" -n 9
            
            ${QEMU_CMD} -L /usr/riscv64-linux-gnu \
                -cpu rv64,v=true,vlen=256,elen=64,vext_spec=v1.0 \
                ./build/bin/llama-cli -m ${MODEL_PATH} -p "What is 200*9?" -n 20
                
            ${QEMU_CMD} -L /usr/riscv64-linux-gnu \
                -cpu rv64,v=true,vlen=256,elen=64,vext_spec=v1.0 \
                ./build/bin/llama-cli -m ${MODEL_PATH} -p "Name three colors" -n 9
                
            ${QEMU_CMD} -L /usr/riscv64-linux-gnu \
                -cpu rv64,v=true,vlen=256,elen=64,vext_spec=v1.0 \
                ./build/bin/llama-cli -m ${MODEL_PATH} -p "What is the average air speed of a swallow?" -n 30
        '''
    }
}
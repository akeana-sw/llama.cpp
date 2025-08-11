pipeline {
    agent any
    
    stages {
        stage('Cleanup') {
            steps {
                cleanWs()
            }
        }
        
        stage('Checkout') {
            steps {
                git url: 'https://github.com/akeana-sw/llama.cpp.git', branch: 'master'
            }
        }
        
        stage('Install dependencies') {
            steps {
                sh '''#!/bin/bash
                    sudo apt update || echo "apt update failed, continuing..."
                    sudo apt install -y build-essential cmake git libcurl4-openssl-dev || echo "Some packages may already be installed"
                '''
            }
        }
        
        stage('Compile') {
            steps {
                sh '''#!/bin/bash
                    mkdir -p build
                    cd build
                    cmake .. -DGGML_RVV=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
                    make -j8 llama-cli
                '''
            }
        }
        
        stage('Test') {
            steps {
                sh '''#!/bin/bash
                    MODEL_PATH="/home/swuser/work/llama.cpp/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
                    
                    ./build/bin/llama-cli -m ${MODEL_PATH} -p "Hello" -n 12
                    ./build/bin/llama-cli -m ${MODEL_PATH} -p "What is 200*9?" -n 30
                    ./build/bin/llama-cli -m ${MODEL_PATH} -p "Name three colors" -n 12
                    ./build/bin/llama-cli -m ${MODEL_PATH} -p "What is the average air speed velocity of a swallow?" -n 30
                '''
            }
        }
    }
}
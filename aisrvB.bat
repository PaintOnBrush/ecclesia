set llamasrvhost=127.0.0.1
set llamasrvport=8081
set llamasrvmodel=c:\models\gemma-3-4b-it-Q4_K_M.gguf
%userprofile%\workspace\llama.cpp\build\bin\Release\llama-server.exe -m %llamasrvmodel% --gpu-layers 14 --host %llamasrvhost% --port %llamasrvport% 

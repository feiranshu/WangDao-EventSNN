savedcmd_/home/amd-edf/hello_module/hello.mod := printf '%s\n'   hello.o | awk '!x[$$0]++ { print("/home/amd-edf/hello_module/"$$0) }' > /home/amd-edf/hello_module/hello.mod

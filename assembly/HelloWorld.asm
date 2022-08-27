.text
.global _start             // Provide program starting address to linker
.align 2                   // Programs have to start on a 64-bit boundary on MacOS

// Setup the parameters to print hello world
// and then call the OS to do it.

_start:
  mov X3, #0

print:
  mov   X0, #1          // 1 = StdOut
  adr   X1, helloworld  // string to print
  mov   X2, #13         // length of our string
  mov   X16, #4         // MacOS write system call
  svc   0               // Call linux to output the string
  add   X3, X3, #1
  cmp   X3, #3
  b.ne  print

// Setup the parameters to exit the program
// and then call the OS to do it.

exit:
  mov     X0, #0        // Use 0 return code
  mov     X16, #1       // Service command code 1 terminates this program
  svc     0             // Call MacOS to terminate the program

helloworld:      .ascii  "Hello World!\n"

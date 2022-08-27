.text
.global _start
.align 2

_start:
  mov   X5, #10
  mov   X3, #0xBEEF
  movk  X3, #0xDEAD, lsl #16
  adrp  X1, output@PAGE
  add   X1, X1, output@PAGEOFF
  adrp  X2, hexstr@PAGE
  add   X2, X2, hexstr@PAGEOFF

loop:
  sub   X5, X5, #1
  and   X4, X3, #0xF
  ldrb  W6, [X2,X4]
  strb  W6, [X1,X5]
  lsr   X3, X3, #4
  cmp   X5, #2
  b.gt  loop

print:
  mov   X0, #1         // 1 = StdOut
  mov   X2, #11        // length of our string
  mov   X16, #4        // MacOS write system call
  svc   0              // Call linux to output the string

exit:
  mov   X0, #0         // Use 0 return code
  mov   X16, #1        // Service command code 1 terminates this program
  svc   0              // Call MacOS to terminate the program


.data
hexstr:      .ascii  "0123456789ABCDEF"
output:      .ascii "0x00000000\n"

.text
.global _start
.align 2

_start:
  adrp  X1, input@PAGE
  add   X1, X1, input@PAGEOFF
  mov   X2, #-1

loop:
  add   X2, X2, #1
  ldrb  W3, [X1,X2]
  cmp   W3, #0
  b.eq  print
  cmp   W3, #('a')
  b.lt  loop
  cmp   W3, #('z')
  b.gt  loop
  sub   W3, W3, #('a'-'A')
  strb  W3, [X1,X2]
  b     loop

print:
  mov   X0, #1         // 1 = StdOut
  mov   X16, #4        // MacOS write system call
  svc   0              // Call linux to output the string

exit:
  mov   X0, #0         // Use 0 return code
  mov   X16, #1        // Service command code 1 terminates this program
  svc   0              // Call MacOS to terminate the program


.data
input:       .ascii  "The quick brown Fox jumps over the lazy Dog\n"

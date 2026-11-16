# Copyright (c) anneal developers
/* Flat blob copied into a low executable page. Rip-relative data. */
.section .text
.globl blob
.globl energy_lj
.globl energy_gupta
.globl grad_enter_lj
.globl grad_exit_lj
.globl grad_enter_gupta
.globl grad_exit_gupta
.globl seed_stub
.globl acct_energy

blob:

/* sub $0x28 keeps the call aligned. r11 is a live Gupta argument and is
   not used as a scratch register. The start timestamp lives in a global
   because the callee spills over the caller's argument area. */
.macro ENERGY_WRAP slot
    sub $0x28, %rsp
    mov %rax, (%rsp)
    mov %rbx, 8(%rsp)
    mov %rcx, 16(%rsp)
    mov %rdx, 24(%rsp)
    movsd %xmm0, 32(%rsp)
    lfence
    rdtsc
    shl $32, %rdx
    or %rax, %rdx
    mov %rdx, tsc0(%rip)
    mov (%rsp), %rax
    mov 8(%rsp), %rbx
    mov 16(%rsp), %rcx
    mov 24(%rsp), %rdx
    movsd 32(%rsp), %xmm0
    call *\slot(%rip)
    movsd %xmm0, 32(%rsp)
    lfence
    rdtsc
    shl $32, %rdx
    or %rax, %rdx
    sub tsc0(%rip), %rdx
    lea acct_energy(%rip), %r11
    lock add %rdx, 16(%r11)
    lock incq (%r11)
    mov 56(%r11), %rax
    test %rax, %rax
    jz 1f
    cmp %rax, (%r11)
    jb 1f
    movl $1, stop(%rip)
    mov $231, %rax
    xor %edi, %edi
    syscall
1:
    movsd 32(%rsp), %xmm0
    movsd best(%rip), %xmm2
    ucomisd %xmm2, %xmm0
    jae 2f
    movsd %xmm0, best(%rip)
2:
    movsd limit(%rip), %xmm2
    ucomisd %xmm2, %xmm0
    ja 3f
    jp 3f
    movl $1, hit(%rip)
    movl $2, stop(%rip)
    mov $231, %rax
    xor %edi, %edi
    syscall
3:
    movsd 32(%rsp), %xmm0
    mov 24(%rsp), %rdx
    mov 16(%rsp), %rcx
    mov 8(%rsp), %rbx
    mov (%rsp), %rax
    add $0x28, %rsp
    ret
.endm

energy_lj:
    ENERGY_WRAP lj_origin

energy_gupta:
    ENERGY_WRAP gupta_origin

.macro GRAD_ENTER disp, slot
    push %rax
    push %rdx
    lfence
    rdtsc
    shl $32, %rdx
    or %rax, %rdx
    mov %rdx, tsc0(%rip)
    pop %rdx
    pop %rax
    lea \disp(%rsp), %r12
    jmp *\slot(%rip)
.endm

grad_enter_lj:
    GRAD_ENTER -0x98, lj_resume

grad_enter_gupta:
    GRAD_ENTER -0x230, gupta_resume

.macro GRAD_EXIT addend
    add $\addend, %rsp
    push %rax
    push %rdx
    lfence
    rdtsc
    shl $32, %rdx
    or %rax, %rdx
    sub tsc0(%rip), %rdx
    push %r11
    lea acct_energy(%rip), %r11
    lock add %rdx, 16(%r11)
    lock incq 8(%r11)
    pop %r11
    pop %rdx
    pop %rax
    ret
.endm

grad_exit_lj:
    GRAD_EXIT 0x118

grad_exit_gupta:
    GRAD_EXIT 0x2b0

seed_stub:
    mov seedv(%rip), %rax
    ret

/* pwrite(acct_fd, &acct_energy, 64, 0) */
write_acct:
    push %rax
    push %rdi
    push %rsi
    push %rdx
    push %rcx
    push %r10
    push %r11
    mov acct_fd(%rip), %edi
    test %edi, %edi
    jz 1f
    lea acct_energy(%rip), %rsi
    mov $96, %edx
    xor %r10d, %r10d
    mov $18, %eax
    syscall
1:
    pop %r11
    pop %r10
    pop %rcx
    pop %rdx
    pop %rsi
    pop %rdi
    pop %rax
    ret

.balign 8
acct_energy: .quad 0
acct_grad:   .quad 0
acct_pot:    .quad 0
best:        .double 1.0e300
reference:   .double 0
tol:         .double 0.001
limit:       .double 0
budget:      .quad 0          /* offset 56 */
seedv:       .quad 0          /* offset 64 */
hit:         .long 0          /* offset 72 */
stop:        .long 0
tsc0:        .quad 0
acct_fd:     .quad 0
lj_resume:    .quad 0x5ae548
gupta_resume: .quad 0x5aabc8
lj_origin:    .quad 0x5af020
gupta_origin: .quad 0x5acac0
save_rax: .quad 0
save_rbx: .quad 0
save_rcx: .quad 0
save_rdx: .quad 0
save_x0: .double 0
save_x1: .double 0
.globl blob_end
blob_end:

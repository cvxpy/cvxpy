"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# gdb-python script: recover the exact faulting instruction from a SIGILL core.
#
# PYTHONFAULTHANDLER catches the SIGILL, prints a traceback, then re-raises via
# raise(), so the crashing thread's innermost frame is __pthread_kill -- not the
# illegal instruction. The original #UD is still on that thread's stack as a
# "<signal handler called>" (SIGTRAMP) frame; the frame just *older* than it is
# the instruction that actually faulted. This script finds that frame on every
# thread and disassembles it, so the mnemonic and raw bytes of the offending
# opcode land in the log. If no SIGTRAMP frame exists (faulthandler disabled --
# the core is taken directly at the #UD), it disassembles each thread's $pc.
#
# Run as:  gdb -batch python <core> -ex 'source tools/decode_sigill_core.py'
import gdb


def _disasm(pc, label):
    print("\n===== %s =====" % label)
    print("RIP = 0x%x" % pc)
    gdb.execute("info symbol 0x%x" % pc)
    gdb.execute("x/4i 0x%x" % pc)
    gdb.execute("x/16xb 0x%x" % pc)


gdb.execute("set pagination off")
inferior = gdb.selected_inferior()
found_sigtramp = False
for thread in inferior.threads():
    thread.switch()
    try:
        frame = gdb.newest_frame()
    except gdb.error:
        continue
    while frame is not None:
        try:
            is_sigtramp = frame.type() == gdb.SIGTRAMP_FRAME
        except gdb.error:
            is_sigtramp = False
        if is_sigtramp:
            faulting = frame.older()
            if faulting is not None:
                faulting.select()
                _disasm(faulting.pc(), "FAULTING INSTRUCTION (thread %d)" % thread.num)
                print("\n===== BACKTRACE OF FAULTING THREAD =====")
                gdb.execute("bt")
                found_sigtramp = True
            break
        frame = frame.older()
    if found_sigtramp:
        break

if not found_sigtramp:
    print("\nno SIGTRAMP frame found (faulthandler off?); disassembling each $pc:")
    for thread in inferior.threads():
        thread.switch()
        try:
            _disasm(gdb.newest_frame().pc(), "thread %d $pc" % thread.num)
        except gdb.error:
            continue
    print("\n===== ALL-THREAD BACKTRACE =====")
    gdb.execute("thread apply all bt")

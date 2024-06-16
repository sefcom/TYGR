int main() {
  long x = 0x123;  // rbp - 0x10
  long y = 0x456;  // rbp - 0xc
  long z = x + y;  // rbp - 0x8
  long w = x + z;  // rbp - 0x4
  return w;
}

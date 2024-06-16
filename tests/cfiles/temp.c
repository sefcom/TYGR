int main() {
  int x = 0x123;
  int* p = &x;
  int y = *p + 0x456;
  return y;
}

int main(int argc, char** argv) {
  int a = 0x123;
  int c = 0x999;
  int b;
  if (a == c) {
    b = argc;
  } else {
    b = 0x456;
  }
  return b;
}


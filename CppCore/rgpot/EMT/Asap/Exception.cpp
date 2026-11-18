#include "Exception.h"

Exception::Exception(char const *m)
    : message(new stringstream()) {
  *message << m;
}

Exception::Exception(const Exception &ex)
    : message(new stringstream()) {
  *message << ex.message->str();
}

Exception::~Exception() { delete message; }
const char *Exception::GetMessage() const {
  static thread_local std::string cache;
  cache = message->str();
  return cache.c_str();
}

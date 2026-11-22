#include "Exception.h"

Exception::Exception(char const *m) : message(m == nullptr ? "" : m) {}

const std::string &Exception::GetMessage() const noexcept { return message; }

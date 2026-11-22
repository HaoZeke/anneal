#ifndef _EXCEPTION_H
#define _EXCEPTION_H

#include <sstream>
#include <string>

// this Exception class can be used as follows:
//
// throw Exception("Error");
//
// throw Exception("Error ") << n + 7 << " while opening file: " << filename;

class Exception {
public:
  Exception(char const *m);
  Exception(const Exception &) = default;
  Exception(Exception &&) noexcept = default;
  Exception &operator=(const Exception &) = default;
  Exception &operator=(Exception &&) noexcept = default;
  ~Exception() = default;
  template <class T> Exception &operator<<(const T &x) {
    std::ostringstream out;
    out << x;
    message += out.str();
    return *this;
  }
  const std::string &GetMessage() const noexcept;

private:
  std::string message;
};

#endif // _EXCEPTION_H

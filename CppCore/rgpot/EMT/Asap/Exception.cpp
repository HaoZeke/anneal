#include "Exception.h"

#include <string>
#include <utility>

namespace {

// The public member stays a `stringstream *`, so sizeof(Exception) and the
// inline operator<< of existing consumers are unchanged. The pointee is this
// type, which adds the buffer that GetMessage hands out. Only this file
// allocates it.
struct MessageStream : std::stringstream {
  std::string text;
};

MessageStream *buffer(stringstream *message) {
  return static_cast<MessageStream *>(message);
}

} // namespace

Exception::Exception(char const *m)
    : message(new MessageStream()) {
  if (m != nullptr) {
    *message << m;
  }
}

Exception::Exception(const Exception &ex)
    : message(new MessageStream()) {
  *message << ex.message->str();
}

Exception &Exception::operator=(const Exception &ex) {
  if (this != &ex) {
    const std::string copied = ex.message->str();
    message->clear();
    message->str(std::string());
    *message << copied;
  }
  return *this;
}

Exception::~Exception() { delete buffer(message); }

const char *Exception::GetMessage() const noexcept {
  MessageStream *owner = buffer(message);
  try {
    std::string current = owner->str();
    if (current != owner->text) {
      owner->text = std::move(current);
    }
  } catch (...) {
    return owner->text.c_str();
  }
  return owner->text.c_str();
}

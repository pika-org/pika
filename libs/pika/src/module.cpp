// This is the public pika module. It reexports only public functionality.

export module pika;

// TODO: This exports as well??
import pika.assertion;

// using is put in given namespace, export must fully qualify reexported names
namespace pika::detail {
export using ::pika::detail::handle_assert;
export using ::pika::detail::set_assertion_handler;
export using ::pika::detail::source_location;
} // namespace pika::detail

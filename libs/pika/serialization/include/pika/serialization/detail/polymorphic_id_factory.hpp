//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/modules/errors.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/serialization/detail/polymorphic_intrusive_factory.hpp>
#include <pika/serialization/serialization_fwd.hpp>
#include <pika/serialization/traits/polymorphic_traits.hpp>
#include <pika/type_support/static.hpp>
#include <pika/type_support/unused.hpp>

#include <cstdint>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include <pika/local/config/warnings_prefix.hpp>

namespace pika { namespace serialization { namespace detail {
    class id_registry
    {
    public:
        PIKA_NON_COPYABLE(id_registry);

    public:
        typedef void* (*ctor_t)();
        typedef std::map<std::string, ctor_t> typename_to_ctor_t;
        typedef std::map<std::string, std::uint32_t> typename_to_id_t;
        typedef std::vector<ctor_t> cache_t;

        static constexpr std::uint32_t invalid_id = ~0u;

        PIKA_EXPORT void register_factory_function(
            const std::string& type_name, ctor_t ctor);

        PIKA_EXPORT void register_typename(
            const std::string& type_name, std::uint32_t id);

        PIKA_EXPORT void fill_missing_typenames();

        PIKA_EXPORT std::uint32_t try_get_id(
            const std::string& type_name) const;

        std::uint32_t get_max_registered_id() const
        {
            return max_id;
        }

        PIKA_EXPORT std::vector<std::string> get_unassigned_typenames()
            const;

        PIKA_EXPORT static id_registry& instance();

    private:
        id_registry()
          : max_id(0u)
        {
        }

        friend struct ::pika::util::static_<id_registry>;
        friend class polymorphic_id_factory;

        PIKA_EXPORT void cache_id(std::uint32_t id, ctor_t ctor);

        std::uint32_t max_id;
        typename_to_ctor_t typename_to_ctor;
        typename_to_id_t typename_to_id;
        cache_t cache;
    };

    class polymorphic_id_factory
    {
    public:
        PIKA_NON_COPYABLE(polymorphic_id_factory);

    private:
        typedef id_registry::ctor_t ctor_t;
        typedef id_registry::typename_to_ctor_t typename_to_ctor_t;
        typedef id_registry::typename_to_id_t typename_to_id_t;
        typedef id_registry::cache_t cache_t;

    public:
        template <class T>
        static T* create(std::uint32_t id, std::string const* name = nullptr)
        {
            const cache_t& vec = id_registry::instance().cache;

            if (id >= vec.size())    //-V104
            {
                std::string msg(
                    "Unknown type descriptor " + std::to_string(id));
#if defined(PIKA_DEBUG)
                if (name != nullptr)
                {
                    msg += ", for typename " + *name + "\n";
                    msg += collect_registered_typenames();
                }
#else
                PIKA_UNUSED(name);
#endif
                PIKA_THROW_EXCEPTION(
                    serialization_error, "polymorphic_id_factory::create", msg);
            }

            ctor_t ctor = vec[id];
            PIKA_ASSERT(ctor != nullptr);    //-V108
            return static_cast<T*>(ctor());
        }

        PIKA_EXPORT static std::uint32_t get_id(
            const std::string& type_name);

    private:
        polymorphic_id_factory() {}

        PIKA_EXPORT static polymorphic_id_factory& instance();
        PIKA_EXPORT static std::string collect_registered_typenames();

        friend struct pika::util::static_<polymorphic_id_factory>;
    };

    template <class T>
    struct register_class_name<T,
        typename std::enable_if<traits::is_serialized_with_id<T>::value>::type>
    {
        register_class_name()
        {
            id_registry::instance().register_factory_function(
                T::pika_serialization_get_name_impl(), &factory_function);
        }

        static void* factory_function()
        {
            return new T;
        }

        register_class_name& instantiate()
        {
            return *this;
        }

        static register_class_name instance;
    };

    template <class T>
    register_class_name<T,
        typename std::enable_if<traits::is_serialized_with_id<T>::value>::type>
        register_class_name<T,
            typename std::enable_if<
                traits::is_serialized_with_id<T>::value>::type>::instance;

    template <std::uint32_t desc>
    std::string get_constant_entry_name();

    template <std::uint32_t Id>
    struct add_constant_entry
    {
        add_constant_entry()
        {
            id_registry::instance().register_typename(
                get_constant_entry_name<Id>(), Id);
        }

        static add_constant_entry instance;
    };

    template <std::uint32_t Id>
    add_constant_entry<Id> add_constant_entry<Id>::instance;

}}}    // namespace pika::serialization::detail

#include <pika/local/config/warnings_suffix.hpp>

#define PIKA_SERIALIZATION_ADD_CONSTANT_ENTRY(String, Id)                       \
    namespace pika { namespace serialization { namespace detail {               \
                template <>                                                    \
                std::string get_constant_entry_name<Id>()                      \
                {                                                              \
                    return PIKA_PP_STRINGIZE(String);                           \
                }                                                              \
                template add_constant_entry<Id>                                \
                    add_constant_entry<Id>::instance;                          \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /**/

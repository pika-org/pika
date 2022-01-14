//  Copyright (c) 2014 Thomas Heller
//  Copyright (c) 2015 Anton Bikineev
//  Copyright (c) 2015 Andreas Schaefer
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/assert.hpp>
#include <pika/modules/debugging.hpp>
#include <pika/modules/errors.hpp>
#include <pika/modules/hashing.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/preprocessor/strip_parens.hpp>
#include <pika/serialization/detail/non_default_constructible.hpp>
#include <pika/serialization/serialization_fwd.hpp>
#include <pika/serialization/traits/needs_automatic_registration.hpp>
#include <pika/serialization/traits/polymorphic_traits.hpp>
#include <pika/type_support/static.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>

#include <pika/local/config/warnings_prefix.hpp>

namespace pika { namespace serialization { namespace detail {

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    struct get_serialization_name
#ifdef PIKA_DISABLE_AUTOMATIC_SERIALIZATION_REGISTRATION
        ;
#else
    {
        const char* operator()()
        {
            /// If you encounter this assert while compiling code, that means that
            /// you have a PIKA_REGISTER_ACTION macro somewhere in a source file,
            /// but the header in which the action is defined misses a
            /// PIKA_REGISTER_ACTION_DECLARATION
            static_assert(traits::needs_automatic_registration<T>::value,
                "PIKA_REGISTER_ACTION_DECLARATION missing");
            return util::debug::type_id<T>::typeid_.type_id();
        }
    };
#endif

    struct function_bunch_type
    {
        using save_function_type = void (*)(output_archive&, void const* base);
        using load_function_type = void (*)(input_archive&, void* base);
        using create_function_type = void* (*) (input_archive&);

        save_function_type save_function;
        load_function_type load_function;
        create_function_type create_function;
    };

    template <typename T>
    class constructor_selector_ptr
    {
    public:
        static T* create(input_archive& ar)
        {
            return create(ar, std::is_default_constructible<T>());
        }

        // is default-constructible
        static T* create(input_archive& ar, std::true_type)
        {
            T* t = new T;
            try
            {
                load_polymorphic(
                    t, ar, pika::traits::is_nonintrusive_polymorphic<T>());
            }
            catch (...)
            {
                delete t;
                throw;
            }
            return t;
        }

        // is non-default-constructible
        static T* create(input_archive& ar, std::false_type)
        {
            using storage_type =
                typename std::aligned_storage<sizeof(T), alignof(T)>::type;

            storage_type* storage = new storage_type;
            T* t = reinterpret_cast<T*>(storage);
            load_construct_data(ar, t, 0);

            try
            {
                load_polymorphic(
                    t, ar, pika::traits::is_nonintrusive_polymorphic<T>());
            }
            catch (...)
            {
                delete t;
                throw;
            }
            return t;
        }

    private:
        static void load_polymorphic(T* t, input_archive& ar, std::true_type)
        {
            serialize(ar, *t, 0);
        }

        static void load_polymorphic(T* t, input_archive& ar, std::false_type)
        {
            ar >> *t;
        }
    };

    class polymorphic_nonintrusive_factory
    {
    public:
        PIKA_NON_COPYABLE(polymorphic_nonintrusive_factory);

    public:
        using serializer_map_type = std::unordered_map<std::string,
            function_bunch_type, pika::util::jenkins_hash>;
        using serializer_typeinfo_map_type = std::unordered_map<std::string,
            std::string, pika::util::jenkins_hash>;

        PIKA_EXPORT static polymorphic_nonintrusive_factory& instance();

        void register_class(std::type_info const& typeinfo,
            std::string const& class_name, function_bunch_type const& bunch)
        {
            if (!typeinfo.name() && std::string(typeinfo.name()).empty())
            {
                PIKA_THROW_EXCEPTION(serialization_error,
                    "polymorphic_nonintrusive_factory::register_class",
                    "Cannot register a factory with an empty type name");
            }
            if (class_name.empty())
            {
                PIKA_THROW_EXCEPTION(serialization_error,
                    "polymorphic_nonintrusive_factory::register_class",
                    "Cannot register a factory with an empty name");
            }
            auto it = map_.find(class_name);
            auto jt = typeinfo_map_.find(typeinfo.name());

            if (it == map_.end())
                map_[class_name] = bunch;
            if (jt == typeinfo_map_.end())
                typeinfo_map_[typeinfo.name()] = class_name;
        }

        // the following templates are defined in *.ipp file
        template <typename T>
        void save(output_archive& ar, const T& t);

        template <typename T>
        void load(input_archive& ar, T& t);

        // use raw pointer to construct either
        // shared_ptr or intrusive_ptr from it
        template <typename T>
        T* load(input_archive& ar);

    private:
        polymorphic_nonintrusive_factory() {}

        friend struct pika::util::static_<polymorphic_nonintrusive_factory>;

        serializer_map_type map_;
        serializer_typeinfo_map_type typeinfo_map_;
    };

    template <typename Derived>
    struct register_class
    {
        static void save(output_archive& ar, const void* base)
        {
            serialize(ar, *static_cast<Derived*>(const_cast<void*>(base)), 0);
        }

        static void load(input_archive& ar, void* base)
        {
            serialize(ar, *static_cast<Derived*>(base), 0);
        }

        // this function is needed for pointer type serialization
        static void* create(input_archive& ar)
        {
            return constructor_selector_ptr<Derived>::create(ar);
        }

        register_class()
        {
            function_bunch_type bunch = {&register_class<Derived>::save,
                &register_class<Derived>::load,
                &register_class<Derived>::create};

            // It's safe to call typeid here. The typeid(t) return value is
            // only used for local lookup to the portable string that goes over the
            // wire
            polymorphic_nonintrusive_factory::instance().register_class(
                typeid(Derived), get_serialization_name<Derived>()(), bunch);
        }

        static register_class instance;
    };

    template <class T>
    register_class<T> register_class<T>::instance;

}}}    // namespace pika::serialization::detail

#include <pika/local/config/warnings_suffix.hpp>

#define PIKA_SERIALIZATION_REGISTER_CLASS_DECLARATION(Class)                    \
    namespace pika { namespace serialization { namespace detail {               \
                template <>                                                    \
                struct PIKA_ALWAYS_EXPORT get_serialization_name<Class>;  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    namespace pika { namespace traits {                                         \
            template <>                                                        \
            struct needs_automatic_registration<action> : std::false_type      \
            {                                                                  \
            };                                                                 \
        }                                                                      \
    }                                                                          \
    PIKA_TRAITS_NONINTRUSIVE_POLYMORPHIC(Class)                                 \
/**/
#define PIKA_SERIALIZATION_REGISTER_CLASS_NAME(Class, Name)                     \
    namespace pika { namespace serialization { namespace detail {               \
                template <>                                                    \
                struct PIKA_ALWAYS_EXPORT get_serialization_name<Class>   \
                {                                                              \
                    char const* operator()()                                   \
                    {                                                          \
                        return Name;                                           \
                    }                                                          \
                };                                                             \
        }}                                                                     \
    }                                                                          \
    template pika::serialization::detail::register_class<Class>                 \
        pika::serialization::detail::register_class<Class>::instance;           \
/**/
#define PIKA_SERIALIZATION_REGISTER_CLASS_NAME_TEMPLATE(                        \
    Parameters, Template, Name)                                                \
    namespace pika { namespace serialization { namespace detail {               \
                PIKA_PP_STRIP_PARENS(Parameters)                                \
                struct PIKA_ALWAYS_EXPORT                                 \
                    get_serialization_name<PIKA_PP_STRIP_PARENS(Template)>      \
                {                                                              \
                    char const* operator()()                                   \
                    {                                                          \
                        return Name;                                           \
                    }                                                          \
                };                                                             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
/**/
#define PIKA_SERIALIZATION_REGISTER_CLASS(Class)                                \
    PIKA_SERIALIZATION_REGISTER_CLASS_NAME(Class, PIKA_PP_STRINGIZE(Class))      \
/**/
#define PIKA_SERIALIZATION_REGISTER_CLASS_TEMPLATE(Parameters, Template)        \
    PIKA_SERIALIZATION_REGISTER_CLASS_NAME_TEMPLATE(Parameters, Template,       \
        pika::util::debug::type_id<PIKA_PP_STRIP_PARENS(Template)>::typeid_      \
            .type_id())                                                        \
    PIKA_PP_STRIP_PARENS(Parameters)                                            \
    pika::serialization::detail::register_class<PIKA_PP_STRIP_PARENS(Template)>  \
        PIKA_PP_STRIP_PARENS(Template)::pika_register_class_instance;            \
/**/
#define PIKA_SERIALIZATION_POLYMORPHIC_TEMPLATE_SEMIINTRUSIVE(Template)         \
    static pika::serialization::detail::register_class<Template>                \
        pika_register_class_instance;                                           \
                                                                               \
    virtual pika::serialization::detail::register_class<Template>&              \
    pika_get_register_class_instance(                                           \
        pika::serialization::detail::register_class<Template>*) const           \
    {                                                                          \
        return pika_register_class_instance;                                    \
    }                                                                          \
/**/
#define PIKA_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR(Class, Func)                 \
    namespace pika { namespace serialization { namespace detail {               \
                template <>                                                    \
                class constructor_selector_ptr<PIKA_PP_STRIP_PARENS(Class)>     \
                {                                                              \
                public:                                                        \
                    static Class* create(input_archive& ar)                    \
                    {                                                          \
                        return Func(ar);                                       \
                    }                                                          \
                };                                                             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
/**/
#define PIKA_SERIALIZATION_WITH_CUSTOM_CONSTRUCTOR_TEMPLATE(                    \
    Parameters, Template, Func)                                                \
    namespace pika { namespace serialization { namespace detail {               \
                PIKA_PP_STRIP_PARENS(Parameters)                                \
                class constructor_selector_ptr<PIKA_PP_STRIP_PARENS(Template)>  \
                {                                                              \
                public:                                                        \
                    static PIKA_PP_STRIP_PARENS(Template) *                     \
                        create(input_archive& ar)                              \
                    {                                                          \
                        return Func(ar,                                        \
                            static_cast<PIKA_PP_STRIP_PARENS(Template)*>(0));   \
                    }                                                          \
                };                                                             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
/**/

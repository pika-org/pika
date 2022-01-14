//  Copyright (c) 2014 Anton Bikineev
//  Copyright (c) 2014 Thomas Heller
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0.
//  See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/local/config.hpp>
#include <pika/modules/debugging.hpp>
#include <pika/modules/hashing.hpp>
#include <pika/preprocessor/stringize.hpp>
#include <pika/serialization/serialization_fwd.hpp>

#include <string>
#include <unordered_map>

namespace pika { namespace serialization { namespace detail {

    class polymorphic_intrusive_factory
    {
    public:
        PIKA_NON_COPYABLE(polymorphic_intrusive_factory);

    private:
        using ctor_type = void* (*) ();
        using ctor_map_type =
            std::unordered_map<std::string, ctor_type, pika::util::jenkins_hash>;

    public:
        polymorphic_intrusive_factory() {}

        PIKA_EXPORT static polymorphic_intrusive_factory& instance();

        PIKA_EXPORT void register_class(
            std::string const& name, ctor_type fun);

        PIKA_EXPORT void* create(std::string const& name) const;

        template <typename T>
        T* create(std::string const& name) const
        {
            return static_cast<T*>(create(name));
        }

    private:
        ctor_map_type map_;
    };

    template <typename T, typename Enable = void>
    struct register_class_name
    {
        register_class_name()
        {
            polymorphic_intrusive_factory::instance().register_class(
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

    template <typename T, typename Enable>
    register_class_name<T, Enable> register_class_name<T, Enable>::instance;

}}}    // namespace pika::serialization::detail

#define PIKA_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name)         \
    template <typename, typename>                                              \
    friend struct ::pika::serialization::detail::register_class_name;           \
                                                                               \
    static std::string pika_serialization_get_name_impl()                       \
    {                                                                          \
        pika::serialization::detail::register_class_name<Class>::instance       \
            .instantiate();                                                    \
        return Name;                                                           \
    }                                                                          \
    virtual std::string pika_serialization_get_name() const                     \
    {                                                                          \
        return Class::pika_serialization_get_name_impl();                       \
    }                                                                          \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, Name)                   \
    PIKA_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);            \
    virtual void load(pika::serialization::input_archive& ar, unsigned n)       \
    {                                                                          \
        serialize<pika::serialization::input_archive>(ar, n);                   \
    }                                                                          \
    virtual void save(pika::serialization::output_archive& ar, unsigned n)      \
        const                                                                  \
    {                                                                          \
        const_cast<Class*>(this)                                               \
            ->serialize<pika::serialization::output_archive>(ar, n);            \
    }                                                                          \
    PIKA_SERIALIZATION_SPLIT_MEMBER()                                           \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(Class, Name)          \
    PIKA_SERIALIZATION_ADD_INTRUSIVE_MEMBERS_WITH_NAME(Class, Name);            \
    virtual void load(pika::serialization::input_archive& ar, unsigned n)       \
    {                                                                          \
        load<pika::serialization::input_archive>(ar, n);                        \
    }                                                                          \
    virtual void save(pika::serialization::output_archive& ar, unsigned n)      \
        const                                                                  \
    {                                                                          \
        save<pika::serialization::output_archive>(ar, n);                       \
    }                                                                          \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC_ABSTRACT(Class)                          \
    virtual std::string pika_serialization_get_name() const = 0;                \
    virtual void load(pika::serialization::input_archive& ar, unsigned n)       \
    {                                                                          \
        serialize<pika::serialization::input_archive>(ar, n);                   \
    }                                                                          \
    virtual void save(pika::serialization::output_archive& ar, unsigned n)      \
        const                                                                  \
    {                                                                          \
        const_cast<Class*>(this)                                               \
            ->serialize<pika::serialization::output_archive>(ar, n);            \
    }                                                                          \
    PIKA_SERIALIZATION_SPLIT_MEMBER()                                           \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC_ABSTRACT_SPLITTED(Class)                 \
    virtual std::string pika_serialization_get_name() const = 0;                \
    virtual void load(pika::serialization::input_archive& ar, unsigned n)       \
    {                                                                          \
        load<pika::serialization::input_archive>(ar, n);                        \
    }                                                                          \
    virtual void save(pika::serialization::output_archive& ar, unsigned n)      \
        const                                                                  \
    {                                                                          \
        save<pika::serialization::output_archive>(ar, n);                       \
    }                                                                          \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC(Class)                                   \
    PIKA_SERIALIZATION_POLYMORPHIC_WITH_NAME(Class, PIKA_PP_STRINGIZE(Class))    \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC_SPLITTED(Class)                          \
    PIKA_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                          \
        Class, PIKA_PP_STRINGIZE(Class))                                        \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC_TEMPLATE(Class)                          \
    PIKA_SERIALIZATION_POLYMORPHIC_WITH_NAME(                                   \
        Class, pika::util::debug::type_id<Class>::typeid_.type_id();)           \
    /**/

#define PIKA_SERIALIZATION_POLYMORPHIC_TEMPLATE_SPLITTED(Class)                 \
    PIKA_SERIALIZATION_POLYMORPHIC_WITH_NAME_SPLITTED(                          \
        Class, pika::util::debug::type_id<T>::typeid_.type_id();)               \
    /**/

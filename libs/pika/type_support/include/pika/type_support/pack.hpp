//  Copyright (c) 2014-2020 Agustin Berge
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <pika/config.hpp>

#include <cstddef>
#include <type_traits>

namespace pika::util::detail {

    template <typename... Ts>
    struct pack
    {
        using type = pack;
        static constexpr std::size_t size = sizeof...(Ts);
    };

    template <typename T, T... Vs>
    struct pack_c
    {
        using type = pack_c;
        static constexpr std::size_t size = sizeof...(Vs);
    };

    template <std::size_t... Is>
    using index_pack = pack_c<std::size_t, Is...>;

    template <typename Left, typename Right>
    struct make_index_pack_join;

    template <std::size_t... Left, std::size_t... Right>
    struct make_index_pack_join<index_pack<Left...>, index_pack<Right...>>
      : index_pack<Left..., (sizeof...(Left) + Right)...>
    {
    };

    template <typename... Ts>
    using decayed_pack = pack<std::decay_t<Ts>...>;

#define PIKA_MAKE_INDEX_PACK_INTEGER_PACK                                                          \
    template <std::size_t N>                                                                       \
    struct make_index_pack : index_pack<__integer_pack(N)...>                                      \
    {                                                                                              \
    };

#define PIKA_MAKE_INDEX_PACK_MAKE_INTEGER_SEQ                                                      \
    template <std::size_t N>                                                                       \
    struct make_index_pack : __make_integer_seq<pack_c, std::size_t, N>                            \
    {                                                                                              \
    };

#define PIKA_MAKE_INDEX_PACK_FALLBACK                                                              \
    template <std::size_t N>                                                                       \
    struct make_index_pack                                                                         \
      : make_index_pack_join<typename make_index_pack<N / 2>::type,                                \
            typename make_index_pack<N - N / 2>::type>                                             \
    {                                                                                              \
    };

#if defined(__has_builtin)
# if __has_builtin(__integer_pack)
    PIKA_MAKE_INDEX_PACK_INTEGER_PACK
# elif __has_builtin(__make_integer_seq)
    PIKA_MAKE_INDEX_PACK_MAKE_INTEGER_SEQ
# else
    PIKA_MAKE_INDEX_PACK_FALLBACK
# endif
#else
    PIKA_MAKE_INDEX_PACK_FALLBACK
#endif

#undef PIKA_MAKE_INDEX_PACK_INTEGER_PACK
#undef PIKA_MAKE_INDEX_PACK_MAKE_INTEGER_SEQ
#undef PIKA_MAKE_INDEX_PACK_FALLBACK

    template <>
    struct make_index_pack<0> : pack_c<std::size_t>
    {
    };

    template <>
    struct make_index_pack<1> : index_pack<0>
    {
    };

    template <std::size_t N>
    using make_index_pack_t = typename make_index_pack<N>::type;

    ///////////////////////////////////////////////////////////////////////////
    // Workaround for clang bug [https://bugs.llvm.org/show_bug.cgi?id=35077]
    template <typename T>
    struct is_true : std::integral_constant<bool, (bool) T::value>
    {
    };

    template <typename T>
    inline constexpr bool is_true_v = is_true<T>::value;

    template <typename T>
    struct is_false : std::integral_constant<bool, !(bool) T::value>
    {
    };

    template <typename T>
    inline constexpr bool is_false_v = is_false<T>::value;

    ///////////////////////////////////////////////////////////////////////////
    template <typename... Ts>
    struct always_true : std::true_type
    {
    };

    template <typename... Ts>
    struct always_false : std::false_type
    {
    };

    template <typename... Ts>
    std::false_type all_of_impl(...);

    template <typename... Ts>
    auto all_of_impl(int) -> always_true<typename std::enable_if_t<is_true_v<Ts>>...>;

    template <typename... Ts>
    struct all_of : decltype(all_of_impl<Ts...>(0))
    {
    };

    template <>
    struct all_of<>    // <fake-type>
      : std::true_type
    {
    };

    template <typename... Ts>
    inline constexpr bool all_of_v = all_of<Ts...>::value;

    template <typename... Ts>
    std::true_type any_of_impl(...);

    template <typename... Ts>
    auto any_of_impl(int) -> always_false<std::enable_if_t<is_false_v<Ts>>...>;

    template <typename... Ts>
    struct any_of : decltype(any_of_impl<Ts...>(0))
    {
    };

    template <>
    struct any_of<>    // <fake-type>
      : std::false_type
    {
    };

    template <typename... Ts>
    inline constexpr bool any_of_v = any_of<Ts...>::value;

    template <typename... Ts>
    struct none_of : std::integral_constant<bool, !any_of_v<Ts...>>
    {
    };

    template <typename... Ts>
    inline constexpr bool none_of_v = none_of<Ts...>::value;

    template <typename T, typename... Ts>
    struct contains : any_of<std::is_same<T, Ts>...>
    {
    };

    struct empty
    {
    };

#define PIKA_AT_INDEX_IMPL_TYPE_PACK_ELEMENT                                                       \
    template <std::size_t I, typename Ts, bool InBounds = (I < Ts::size)>                          \
    struct at_index_impl : empty                                                                   \
    {                                                                                              \
    };                                                                                             \
                                                                                                   \
    template <std::size_t I, typename... Ts>                                                       \
    struct at_index_impl<I, pack<Ts...>, /*InBounds*/ true>                                        \
    {                                                                                              \
        using type = __type_pack_element<I, Ts...>;                                                \
    };

#define PIKA_AT_INDEX_IMPL_FALLBACK                                                                \
    template <std::size_t I, typename T>                                                           \
    struct indexed                                                                                 \
    {                                                                                              \
        using type = T;                                                                            \
    };                                                                                             \
                                                                                                   \
    template <typename Ts, typename Is>                                                            \
    struct indexer;                                                                                \
                                                                                                   \
    template <typename... Ts, std::size_t... Is>                                                   \
    struct indexer<pack<Ts...>, pack_c<std::size_t, Is...>> : indexed<Is, Ts>...                   \
    {                                                                                              \
    };                                                                                             \
                                                                                                   \
    template <std::size_t J>                                                                       \
    empty at_index_check(...);                                                              \
                                                                                                   \
    template <std::size_t J, typename T>                                                           \
    indexed<J, T> at_index_check(indexed<J, T> const&);                                     \
                                                                                                   \
    template <std::size_t I, typename Ts>                                                          \
    struct at_index_impl                                                                           \
      : decltype(detail::at_index_check<I>(                                                        \
            indexer<Ts, typename make_index_pack<Ts::size>::type>()))                              \
    {                                                                                              \
    };

#if defined(__has_builtin)
# if __has_builtin(__type_pack_element)
    PIKA_AT_INDEX_IMPL_TYPE_PACK_ELEMENT
# else
    PIKA_AT_INDEX_IMPL_FALLBACK
# endif
#else
    PIKA_AT_INDEX_IMPL_FALLBACK
#endif

#undef PIKA_AT_INDEX_IMPL_TYPE_PACK_ELEMENT
#undef PIKA_AT_INDEX_IMPL_FALLBACK

    template <std::size_t I, typename... Ts>
    struct at_index : at_index_impl<I, pack<Ts...>>
    {
    };

    template <std::size_t I, typename... Ts>
    using at_index_t = typename at_index<I, Ts...>::type;

    template <typename Pack, template <typename> class Transformer>
    struct transform;

    template <template <typename> class Transformer, template <typename...> class Pack,
        typename... Ts>
    struct transform<Pack<Ts...>, Transformer>
    {
        using type = Pack<typename Transformer<Ts>::type...>;
    };

    /// Apply a meta-function to each element in a pack.
    template <typename Pack, template <typename> class Transformer>
    using transform_t = typename transform<Pack, Transformer>::type;

    template <typename PackUnique, typename PackRest>
    struct unique_helper;

    template <template <typename...> class Pack, typename... Ts>
    struct unique_helper<Pack<Ts...>, Pack<>>
    {
        using type = Pack<Ts...>;
    };

    template <template <typename...> class Pack, typename... Ts, typename U, typename... Us>
    struct unique_helper<Pack<Ts...>, Pack<U, Us...>>
      : std::conditional<contains<U, Ts...>::value, unique_helper<Pack<Ts...>, Pack<Us...>>,
            unique_helper<Pack<Ts..., U>, Pack<Us...>>>::type
    {
    };

    template <typename Pack>
    struct unique;

    template <template <typename...> class Pack, typename... Ts>
    struct unique<Pack<Ts...>> : unique_helper<Pack<>, Pack<Ts...>>
    {
    };

    /// Remove duplicate types in the given pack.
    template <typename Pack>
    using unique_t = typename unique<Pack>::type;

    template <typename... Packs>
    struct concat;

    template <template <typename...> class Pack, typename... Ts>
    struct concat<Pack<Ts...>>
    {
        using type = Pack<Ts...>;
    };

    template <template <typename...> class Pack, typename... Ts, typename... Us, typename... Rest>
    struct concat<Pack<Ts...>, Pack<Us...>, Rest...> : concat<Pack<Ts..., Us...>, Rest...>
    {
    };

    /// Concatenate the elements in the given packs into a single pack. The
    /// packs must be of the same type.
    template <typename... Packs>
    using concat_t = typename concat<Packs...>::type;

    /// Concatenate the elements in the given packs into a single pack and then
    /// remove duplicates.
    template <typename... Packs>
    using unique_concat_t = unique_t<concat_t<Packs...>>;

    template <typename Pack>
    struct concat_pack_of_packs;

    template <template <typename...> class Pack, typename... Ts>
    struct concat_pack_of_packs<Pack<Ts...>>
    {
        using type = typename concat<Ts...>::type;
    };

    /// Concatenate the packs in the given pack into a single pack. The
    /// outer pack is discarded.
    template <typename Pack>
    using concat_pack_of_packs_t = typename concat_pack_of_packs<Pack>::type;

    template <typename Pack>
    struct concat_inner_packs;

    template <template <typename...> class Pack, typename... Ts>
    struct concat_inner_packs<Pack<Ts...>>
    {
        using type = Pack<typename concat<Ts...>::type>;
    };

    /// Concatenate the packs in the given pack into a single pack. The
    /// outer pack is kept.
    template <typename Pack>
    using concat_inner_packs_t = typename concat_inner_packs<Pack>::type;

    template <typename Pack, typename T>
    struct prepend;

    template <typename T, template <typename...> class Pack, typename... Ts>
    struct prepend<Pack<Ts...>, T>
    {
        using type = Pack<T, Ts...>;
    };

    /// Prepend a given type to the given pack.
    template <typename Pack, typename T>
    using prepend_t = typename prepend<Pack, T>::type;

    template <typename Pack, typename T>
    struct append;

    template <typename T, template <typename...> class Pack, typename... Ts>
    struct append<Pack<Ts...>, T>
    {
        using type = Pack<Ts..., T>;
    };

    /// Append a given type to the given pack.
    template <typename Pack, typename T>
    using append_t = typename append<Pack, T>::type;

    template <template <typename...> class NewPack, typename OldPack>
    struct change_pack;

    template <template <typename...> class NewPack, template <typename...> class OldPack,
        typename... Ts>
    struct change_pack<NewPack, OldPack<Ts...>>
    {
        using type = NewPack<Ts...>;
    };

    /// Change a OldPack<Ts...> to NewPack<Ts...>
    template <template <typename...> class NewPack, typename OldPack>
    using change_pack_t = typename change_pack<NewPack, OldPack>::type;
}    // namespace pika::util::detail

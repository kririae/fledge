#include <gtest/gtest.h>

#include <memory>
#include <type_traits>

#include "common/dispatcher.h"
#include "fmt/core.h"

using namespace fledge;

struct A;
struct B;
struct C;
struct D;
struct E;

class Derived : public Dispatcher<A, B> {
public:
  using Dispatcher::Dispatcher;
  virtual ~Derived() = default;

  virtual int funcA(int a, int b);
  virtual int funcA_impl(int a, int b) {
    assert(false);
  }  // should be implemented in derived class
};

struct A : public Derived {
  int funcA_impl(int a, int b) override {
    fmt::print("funcA_impl is called inside A\n");
    return a + b;
  }
};

struct B : public Derived {
  int funcA_impl(int a, int b) override {
    fmt::print("funcA_impl is called inside B\n");
    return a * b;
  }
};

int Derived::funcA(int a, int b) {
  // Capture all the parameters
  auto invoker = [&](auto cls) -> int {
    static_assert(HasType<typename std::pointer_traits<
                              std::remove_cvref_t<decltype(cls)>>::element_type,
                          type_pack>::value);
    return cls->funcA_impl(a, b);
  };  // auto invoker()
  return dispatch(invoker);
}

TEST(Dispatcher, TypePack) {
  using type_pack = TypePack<A, B, C, D>;
  auto pack_size  = type_pack::size;
  EXPECT_EQ(pack_size, 4);

  auto index_a = IndexOf<A, type_pack>::value;
  auto index_b = IndexOf<B, type_pack>::value;
  auto index_c = IndexOf<C, type_pack>::value;
  auto index_d = IndexOf<D, type_pack>::value;
  EXPECT_EQ(index_a, 0);
  EXPECT_EQ(index_b, 1);
  EXPECT_EQ(index_c, 2);
  EXPECT_EQ(index_d, 3);

  auto bool_a = HasType<A, type_pack>::value;
  EXPECT_TRUE(bool_a);
  auto bool_e = HasType<E, type_pack>::value;
  EXPECT_FALSE(bool_e);
}

TEST(Dispatcher, Comprehensive) {
  A      *a          = new A;
  B      *b          = new B;
  Derived dispatch_a = Derived(a);
  Derived dispatch_b = Derived(b);

  int x = 2, y = 3;
  EXPECT_EQ(dispatch_a.funcA(x, y), 5);
  EXPECT_EQ(dispatch_b.funcA(x, y), 6);
}
#include <gtest/gtest.h>

#include <iostream>
#include <memory>

#include "fledge.h"
#include "resource.hpp"

using namespace fledge;

TEST(Resource, Resource) {
  Resource resource;

  struct Type {
    Type() = default;
    Type(int a, int b, int c) : m_a(a), m_b(b), m_c(c) {}
    int m_a, m_b, m_c;

    ~Type() { printf("Type's destructor is called\n"); }
  };

  Type *type_ptr = resource.alloc<Type>(1, 2, 3);
  EXPECT_EQ(type_ptr->m_a, 1);
  EXPECT_EQ(type_ptr->m_b, 2);
  EXPECT_EQ(type_ptr->m_c, 3);

  Type *type_arr = resource.alloc<Type[]>(10, 3, 2, 1);
  EXPECT_EQ(type_arr[9].m_a, 3);
  EXPECT_EQ(type_arr[9].m_b, 2);
  EXPECT_EQ(type_arr[9].m_c, 1);

  int *type_aligned_arr = resource.alignedAlloc<int[], 16>(10);
}

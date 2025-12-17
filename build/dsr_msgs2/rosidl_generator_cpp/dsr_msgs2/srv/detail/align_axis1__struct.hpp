// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/AlignAxis1.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__ALIGN_AXIS1__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__ALIGN_AXIS1__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__AlignAxis1_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__AlignAxis1_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct AlignAxis1_Request_
{
  using Type = AlignAxis1_Request_<ContainerAllocator>;

  explicit AlignAxis1_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->x1.begin(), this->x1.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x2.begin(), this->x2.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x3.begin(), this->x3.end(), 0.0);
      std::fill<typename std::array<double, 3>::iterator, double>(this->source_vect.begin(), this->source_vect.end(), 0.0);
      this->axis = 0;
      this->ref = 0;
    }
  }

  explicit AlignAxis1_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : x1(_alloc),
    x2(_alloc),
    x3(_alloc),
    source_vect(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->x1.begin(), this->x1.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x2.begin(), this->x2.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x3.begin(), this->x3.end(), 0.0);
      std::fill<typename std::array<double, 3>::iterator, double>(this->source_vect.begin(), this->source_vect.end(), 0.0);
      this->axis = 0;
      this->ref = 0;
    }
  }

  // field types and members
  using _x1_type =
    std::array<double, 6>;
  _x1_type x1;
  using _x2_type =
    std::array<double, 6>;
  _x2_type x2;
  using _x3_type =
    std::array<double, 6>;
  _x3_type x3;
  using _source_vect_type =
    std::array<double, 3>;
  _source_vect_type source_vect;
  using _axis_type =
    int8_t;
  _axis_type axis;
  using _ref_type =
    int8_t;
  _ref_type ref;

  // setters for named parameter idiom
  Type & set__x1(
    const std::array<double, 6> & _arg)
  {
    this->x1 = _arg;
    return *this;
  }
  Type & set__x2(
    const std::array<double, 6> & _arg)
  {
    this->x2 = _arg;
    return *this;
  }
  Type & set__x3(
    const std::array<double, 6> & _arg)
  {
    this->x3 = _arg;
    return *this;
  }
  Type & set__source_vect(
    const std::array<double, 3> & _arg)
  {
    this->source_vect = _arg;
    return *this;
  }
  Type & set__axis(
    const int8_t & _arg)
  {
    this->axis = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__AlignAxis1_Request
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__AlignAxis1_Request
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AlignAxis1_Request_ & other) const
  {
    if (this->x1 != other.x1) {
      return false;
    }
    if (this->x2 != other.x2) {
      return false;
    }
    if (this->x3 != other.x3) {
      return false;
    }
    if (this->source_vect != other.source_vect) {
      return false;
    }
    if (this->axis != other.axis) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    return true;
  }
  bool operator!=(const AlignAxis1_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AlignAxis1_Request_

// alias to use template instance with default allocator
using AlignAxis1_Request =
  dsr_msgs2::srv::AlignAxis1_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__AlignAxis1_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__AlignAxis1_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct AlignAxis1_Response_
{
  using Type = AlignAxis1_Response_<ContainerAllocator>;

  explicit AlignAxis1_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit AlignAxis1_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  // field types and members
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__AlignAxis1_Response
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__AlignAxis1_Response
    std::shared_ptr<dsr_msgs2::srv::AlignAxis1_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const AlignAxis1_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const AlignAxis1_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct AlignAxis1_Response_

// alias to use template instance with default allocator
using AlignAxis1_Response =
  dsr_msgs2::srv::AlignAxis1_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct AlignAxis1
{
  using Request = dsr_msgs2::srv::AlignAxis1_Request;
  using Response = dsr_msgs2::srv::AlignAxis1_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__ALIGN_AXIS1__STRUCT_HPP_

// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/EnableAlterMotion.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct EnableAlterMotion_Request_
{
  using Type = EnableAlterMotion_Request_<ContainerAllocator>;

  explicit EnableAlterMotion_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->n = 0l;
      this->mode = 0;
      this->ref = 0;
      std::fill<typename std::array<double, 2>::iterator, double>(this->limit_dpos.begin(), this->limit_dpos.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->limit_dpos_per.begin(), this->limit_dpos_per.end(), 0.0);
    }
  }

  explicit EnableAlterMotion_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : limit_dpos(_alloc),
    limit_dpos_per(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->n = 0l;
      this->mode = 0;
      this->ref = 0;
      std::fill<typename std::array<double, 2>::iterator, double>(this->limit_dpos.begin(), this->limit_dpos.end(), 0.0);
      std::fill<typename std::array<double, 2>::iterator, double>(this->limit_dpos_per.begin(), this->limit_dpos_per.end(), 0.0);
    }
  }

  // field types and members
  using _n_type =
    int32_t;
  _n_type n;
  using _mode_type =
    int8_t;
  _mode_type mode;
  using _ref_type =
    int8_t;
  _ref_type ref;
  using _limit_dpos_type =
    std::array<double, 2>;
  _limit_dpos_type limit_dpos;
  using _limit_dpos_per_type =
    std::array<double, 2>;
  _limit_dpos_per_type limit_dpos_per;

  // setters for named parameter idiom
  Type & set__n(
    const int32_t & _arg)
  {
    this->n = _arg;
    return *this;
  }
  Type & set__mode(
    const int8_t & _arg)
  {
    this->mode = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }
  Type & set__limit_dpos(
    const std::array<double, 2> & _arg)
  {
    this->limit_dpos = _arg;
    return *this;
  }
  Type & set__limit_dpos_per(
    const std::array<double, 2> & _arg)
  {
    this->limit_dpos_per = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Request
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Request
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EnableAlterMotion_Request_ & other) const
  {
    if (this->n != other.n) {
      return false;
    }
    if (this->mode != other.mode) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    if (this->limit_dpos != other.limit_dpos) {
      return false;
    }
    if (this->limit_dpos_per != other.limit_dpos_per) {
      return false;
    }
    return true;
  }
  bool operator!=(const EnableAlterMotion_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EnableAlterMotion_Request_

// alias to use template instance with default allocator
using EnableAlterMotion_Request =
  dsr_msgs2::srv::EnableAlterMotion_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct EnableAlterMotion_Response_
{
  using Type = EnableAlterMotion_Response_<ContainerAllocator>;

  explicit EnableAlterMotion_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit EnableAlterMotion_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Response
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__EnableAlterMotion_Response
    std::shared_ptr<dsr_msgs2::srv::EnableAlterMotion_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const EnableAlterMotion_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const EnableAlterMotion_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct EnableAlterMotion_Response_

// alias to use template instance with default allocator
using EnableAlterMotion_Response =
  dsr_msgs2::srv::EnableAlterMotion_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct EnableAlterMotion
{
  using Request = dsr_msgs2::srv::EnableAlterMotion_Request;
  using Response = dsr_msgs2::srv::EnableAlterMotion_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__ENABLE_ALTER_MOTION__STRUCT_HPP_

// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/IsDoneBoltTightening.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__IS_DONE_BOLT_TIGHTENING__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__IS_DONE_BOLT_TIGHTENING__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct IsDoneBoltTightening_Request_
{
  using Type = IsDoneBoltTightening_Request_<ContainerAllocator>;

  explicit IsDoneBoltTightening_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->m = 0.0;
      this->timeout = 0.0;
      this->axis = 0;
    }
  }

  explicit IsDoneBoltTightening_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    (void)_alloc;
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->m = 0.0;
      this->timeout = 0.0;
      this->axis = 0;
    }
  }

  // field types and members
  using _m_type =
    double;
  _m_type m;
  using _timeout_type =
    double;
  _timeout_type timeout;
  using _axis_type =
    int8_t;
  _axis_type axis;

  // setters for named parameter idiom
  Type & set__m(
    const double & _arg)
  {
    this->m = _arg;
    return *this;
  }
  Type & set__timeout(
    const double & _arg)
  {
    this->timeout = _arg;
    return *this;
  }
  Type & set__axis(
    const int8_t & _arg)
  {
    this->axis = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Request
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Request
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const IsDoneBoltTightening_Request_ & other) const
  {
    if (this->m != other.m) {
      return false;
    }
    if (this->timeout != other.timeout) {
      return false;
    }
    if (this->axis != other.axis) {
      return false;
    }
    return true;
  }
  bool operator!=(const IsDoneBoltTightening_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct IsDoneBoltTightening_Request_

// alias to use template instance with default allocator
using IsDoneBoltTightening_Request =
  dsr_msgs2::srv::IsDoneBoltTightening_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct IsDoneBoltTightening_Response_
{
  using Type = IsDoneBoltTightening_Response_<ContainerAllocator>;

  explicit IsDoneBoltTightening_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->success = false;
    }
  }

  explicit IsDoneBoltTightening_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
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
    dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Response
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__IsDoneBoltTightening_Response
    std::shared_ptr<dsr_msgs2::srv::IsDoneBoltTightening_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const IsDoneBoltTightening_Response_ & other) const
  {
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const IsDoneBoltTightening_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct IsDoneBoltTightening_Response_

// alias to use template instance with default allocator
using IsDoneBoltTightening_Response =
  dsr_msgs2::srv::IsDoneBoltTightening_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct IsDoneBoltTightening
{
  using Request = dsr_msgs2::srv::IsDoneBoltTightening_Request;
  using Response = dsr_msgs2::srv::IsDoneBoltTightening_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__IS_DONE_BOLT_TIGHTENING__STRUCT_HPP_

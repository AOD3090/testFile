// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/Trans.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__TRANS__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__TRANS__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__Trans_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__Trans_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Trans_Request_
{
  using Type = Trans_Request_<ContainerAllocator>;

  explicit Trans_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->delta.begin(), this->delta.end(), 0.0);
      this->ref = 0;
      this->ref_out = 0;
    }
  }

  explicit Trans_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : pos(_alloc),
    delta(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->pos.begin(), this->pos.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->delta.begin(), this->delta.end(), 0.0);
      this->ref = 0;
      this->ref_out = 0;
    }
  }

  // field types and members
  using _pos_type =
    std::array<double, 6>;
  _pos_type pos;
  using _delta_type =
    std::array<double, 6>;
  _delta_type delta;
  using _ref_type =
    int8_t;
  _ref_type ref;
  using _ref_out_type =
    int8_t;
  _ref_out_type ref_out;

  // setters for named parameter idiom
  Type & set__pos(
    const std::array<double, 6> & _arg)
  {
    this->pos = _arg;
    return *this;
  }
  Type & set__delta(
    const std::array<double, 6> & _arg)
  {
    this->delta = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }
  Type & set__ref_out(
    const int8_t & _arg)
  {
    this->ref_out = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::Trans_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::Trans_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::Trans_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::Trans_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__Trans_Request
    std::shared_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__Trans_Request
    std::shared_ptr<dsr_msgs2::srv::Trans_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Trans_Request_ & other) const
  {
    if (this->pos != other.pos) {
      return false;
    }
    if (this->delta != other.delta) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    if (this->ref_out != other.ref_out) {
      return false;
    }
    return true;
  }
  bool operator!=(const Trans_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Trans_Request_

// alias to use template instance with default allocator
using Trans_Request =
  dsr_msgs2::srv::Trans_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__Trans_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__Trans_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct Trans_Response_
{
  using Type = Trans_Response_<ContainerAllocator>;

  explicit Trans_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->trans_pos.begin(), this->trans_pos.end(), 0.0);
      this->success = false;
    }
  }

  explicit Trans_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : trans_pos(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->trans_pos.begin(), this->trans_pos.end(), 0.0);
      this->success = false;
    }
  }

  // field types and members
  using _trans_pos_type =
    std::array<double, 6>;
  _trans_pos_type trans_pos;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__trans_pos(
    const std::array<double, 6> & _arg)
  {
    this->trans_pos = _arg;
    return *this;
  }
  Type & set__success(
    const bool & _arg)
  {
    this->success = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::Trans_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::Trans_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::Trans_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::Trans_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__Trans_Response
    std::shared_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__Trans_Response
    std::shared_ptr<dsr_msgs2::srv::Trans_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const Trans_Response_ & other) const
  {
    if (this->trans_pos != other.trans_pos) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const Trans_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct Trans_Response_

// alias to use template instance with default allocator
using Trans_Response =
  dsr_msgs2::srv::Trans_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct Trans
{
  using Request = dsr_msgs2::srv::Trans_Request;
  using Response = dsr_msgs2::srv::Trans_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__TRANS__STRUCT_HPP_

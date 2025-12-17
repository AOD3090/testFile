// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from dsr_msgs2:srv/CalcCoord.idl
// generated code does not contain a copyright notice

#ifndef DSR_MSGS2__SRV__DETAIL__CALC_COORD__STRUCT_HPP_
#define DSR_MSGS2__SRV__DETAIL__CALC_COORD__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__CalcCoord_Request __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__CalcCoord_Request __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct CalcCoord_Request_
{
  using Type = CalcCoord_Request_<ContainerAllocator>;

  explicit CalcCoord_Request_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->input_pos_cnt = 0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->x1.begin(), this->x1.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x2.begin(), this->x2.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x3.begin(), this->x3.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x4.begin(), this->x4.end(), 0.0);
      this->ref = 0;
      this->mod = 0;
    }
  }

  explicit CalcCoord_Request_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : x1(_alloc),
    x2(_alloc),
    x3(_alloc),
    x4(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->input_pos_cnt = 0;
      std::fill<typename std::array<double, 6>::iterator, double>(this->x1.begin(), this->x1.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x2.begin(), this->x2.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x3.begin(), this->x3.end(), 0.0);
      std::fill<typename std::array<double, 6>::iterator, double>(this->x4.begin(), this->x4.end(), 0.0);
      this->ref = 0;
      this->mod = 0;
    }
  }

  // field types and members
  using _input_pos_cnt_type =
    int8_t;
  _input_pos_cnt_type input_pos_cnt;
  using _x1_type =
    std::array<double, 6>;
  _x1_type x1;
  using _x2_type =
    std::array<double, 6>;
  _x2_type x2;
  using _x3_type =
    std::array<double, 6>;
  _x3_type x3;
  using _x4_type =
    std::array<double, 6>;
  _x4_type x4;
  using _ref_type =
    int8_t;
  _ref_type ref;
  using _mod_type =
    int8_t;
  _mod_type mod;

  // setters for named parameter idiom
  Type & set__input_pos_cnt(
    const int8_t & _arg)
  {
    this->input_pos_cnt = _arg;
    return *this;
  }
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
  Type & set__x4(
    const std::array<double, 6> & _arg)
  {
    this->x4 = _arg;
    return *this;
  }
  Type & set__ref(
    const int8_t & _arg)
  {
    this->ref = _arg;
    return *this;
  }
  Type & set__mod(
    const int8_t & _arg)
  {
    this->mod = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__CalcCoord_Request
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__CalcCoord_Request
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Request_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CalcCoord_Request_ & other) const
  {
    if (this->input_pos_cnt != other.input_pos_cnt) {
      return false;
    }
    if (this->x1 != other.x1) {
      return false;
    }
    if (this->x2 != other.x2) {
      return false;
    }
    if (this->x3 != other.x3) {
      return false;
    }
    if (this->x4 != other.x4) {
      return false;
    }
    if (this->ref != other.ref) {
      return false;
    }
    if (this->mod != other.mod) {
      return false;
    }
    return true;
  }
  bool operator!=(const CalcCoord_Request_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CalcCoord_Request_

// alias to use template instance with default allocator
using CalcCoord_Request =
  dsr_msgs2::srv::CalcCoord_Request_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2


#ifndef _WIN32
# define DEPRECATED__dsr_msgs2__srv__CalcCoord_Response __attribute__((deprecated))
#else
# define DEPRECATED__dsr_msgs2__srv__CalcCoord_Response __declspec(deprecated)
#endif

namespace dsr_msgs2
{

namespace srv
{

// message struct
template<class ContainerAllocator>
struct CalcCoord_Response_
{
  using Type = CalcCoord_Response_<ContainerAllocator>;

  explicit CalcCoord_Response_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->conv_posx.begin(), this->conv_posx.end(), 0.0);
      this->success = false;
    }
  }

  explicit CalcCoord_Response_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : conv_posx(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      std::fill<typename std::array<double, 6>::iterator, double>(this->conv_posx.begin(), this->conv_posx.end(), 0.0);
      this->success = false;
    }
  }

  // field types and members
  using _conv_posx_type =
    std::array<double, 6>;
  _conv_posx_type conv_posx;
  using _success_type =
    bool;
  _success_type success;

  // setters for named parameter idiom
  Type & set__conv_posx(
    const std::array<double, 6> & _arg)
  {
    this->conv_posx = _arg;
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
    dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator> *;
  using ConstRawPtr =
    const dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__dsr_msgs2__srv__CalcCoord_Response
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__dsr_msgs2__srv__CalcCoord_Response
    std::shared_ptr<dsr_msgs2::srv::CalcCoord_Response_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const CalcCoord_Response_ & other) const
  {
    if (this->conv_posx != other.conv_posx) {
      return false;
    }
    if (this->success != other.success) {
      return false;
    }
    return true;
  }
  bool operator!=(const CalcCoord_Response_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct CalcCoord_Response_

// alias to use template instance with default allocator
using CalcCoord_Response =
  dsr_msgs2::srv::CalcCoord_Response_<std::allocator<void>>;

// constant definitions

}  // namespace srv

}  // namespace dsr_msgs2

namespace dsr_msgs2
{

namespace srv
{

struct CalcCoord
{
  using Request = dsr_msgs2::srv::CalcCoord_Request;
  using Response = dsr_msgs2::srv::CalcCoord_Response;
};

}  // namespace srv

}  // namespace dsr_msgs2

#endif  // DSR_MSGS2__SRV__DETAIL__CALC_COORD__STRUCT_HPP_
